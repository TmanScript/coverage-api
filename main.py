import os
import zipfile
import tempfile
from typing import Optional, List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.geocoders import GoogleV3
from bs4 import BeautifulSoup
import json

# ==========================================
# CONFIGURATION
# ==========================================
KMZ_FILE = "towers.kmz"
COVERAGE_RADIUS_KM = 5.0
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# ==========================================
# COVERAGE CHECKER CLASS
# ==========================================
class CoverageChecker:
    def __init__(self, kmz_path):
        print(f"Loading KMZ: {kmz_path}...")
        self.gdf = self.load_kmz_manually(kmz_path)

        if self.gdf.empty:
            print("WARNING: KMZ loaded but contains no data features!")
            self.polygons = gpd.GeoDataFrame()
            self.points = gpd.GeoDataFrame()
        else:
            self.polygons = self.gdf[self.gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
            self.points = self.gdf[self.gdf.geom_type.isin(['Point', 'MultiPoint'])]
            print(f"Total Features: {len(self.gdf)}")

    def parse_coords_string(self, coord_str):
        coords = []
        raw_points = coord_str.strip().split()
        for p in raw_points:
            parts = p.split(',')
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))
        return coords

    def load_kmz_manually(self, kmz_path):
        if not os.path.exists(kmz_path):
            raise FileNotFoundError(f"File {kmz_path} not found.")

        features = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(kmz_path, 'r') as z:
                kml_files = [f for f in z.namelist() if f.endswith('.kml')]
                if not kml_files:
                    return gpd.GeoDataFrame()

                kml_file = kml_files[0]
                z.extract(kml_file, temp_dir)
                full_path = os.path.join(temp_dir, kml_file)

                print("Parsing KML XML directly...")
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f, 'xml')
                    placemarks = soup.find_all('Placemark')

                    for p in placemarks:
                        try:
                            name = p.find('name').text if p.find('name') else "Unknown"
                            desc = p.find('description').text if p.find('description') else ""
                            geometry = None

                            point_tag = p.find('Point')
                            if point_tag:
                                coords_tag = point_tag.find('coordinates')
                                if coords_tag:
                                    coords = self.parse_coords_string(coords_tag.text)
                                    if coords:
                                        geometry = Point(coords[0])

                            poly_tag = p.find('Polygon')
                            if poly_tag:
                                outer = poly_tag.find('outerBoundaryIs')
                                if outer:
                                    coords_tag = outer.find('coordinates')
                                    if coords_tag:
                                        coords = self.parse_coords_string(coords_tag.text)
                                        if len(coords) >= 3:
                                            geometry = Polygon(coords)

                            if geometry:
                                features.append({'name': name, 'description': desc, 'geometry': geometry})
                        except Exception:
                            continue

        if not features:
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(features, crs="EPSG:4326")

    def check_point(self, lat: float, lon: float):
        user_point = Point(lon, lat)

        # POLYGON MATCH
        if not self.polygons.empty:
            matches = self.polygons.contains(user_point)
            if matches.any():
                hit = self.polygons[matches].iloc[0]
                details = {k: v for k, v in hit.to_dict().items() if k != 'geometry'}
                details['match_type'] = 'Inside Polygon Coverage'
                return True, details

        # POINT PROXIMITY
        if not self.points.empty:
            user_point_proj = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs("EPSG:3857")[0]
            points_proj = self.points.to_crs("EPSG:3857")
            distances = points_proj.distance(user_point_proj)

            radius_meters = COVERAGE_RADIUS_KM * 1000
            nearby_mask = distances <= radius_meters

            if nearby_mask.any():
                nearest_idx = distances.idxmin()
                nearest = self.points.iloc[nearest_idx].to_dict()
                dist = distances.min()
                details = {k: v for k, v in nearest.items() if k != 'geometry'}
                details['match_type'] = 'Tower Proximity'
                details['distance_km'] = round(dist / 1000, 2)
                return True, details

        return False, None


# ==========================================
# FASTAPI SETUP
# ==========================================
app = FastAPI(title="Coverage Check API")

checker = None
try:
    checker = CoverageChecker(KMZ_FILE)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load KMZ. {e}")

try:
    geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY, timeout=10)
except:
    geolocator = None


# ==========================================
# MODELS
# ==========================================
class CoverageResponse(BaseModel):
    address: str
    latitude: float
    longitude: float
    in_coverage: bool
    details: Optional[dict] = None


class CoordsRequest(BaseModel):
    latitude: Optional[Union[str, float]] = None
    longitude: Optional[Union[str, float]] = None
    address: Optional[str] = None


# ==========================================
# POST /check — Supports Chatrace JSON strings
# ==========================================
@app.post("/check", response_model=CoverageResponse)
async def check_coverage_json(req: CoordsRequest):
    if not checker:
        raise HTTPException(status_code=500, detail="Coverage map not loaded.")

    # ---------------------------
    # A) FIX FOR CHATRACE STRING JSON
    # ---------------------------
    def parse_geo_string(value):
        """Detect if Chatrace sent a JSON object inside a string."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("{") and value.endswith("}"):
                try:
                    return json.loads(value)
                except:
                    pass
        return None

    # Check latitude or longitude for embedded JSON
    geo_from_lat = parse_geo_string(req.latitude)
    geo_from_lon = parse_geo_string(req.longitude)

    if geo_from_lat:
        req.latitude = geo_from_lat.get("latitude")
        req.longitude = geo_from_lat.get("longitude")

    if geo_from_lon:
        req.latitude = geo_from_lon.get("latitude")
        req.longitude = geo_from_lon.get("longitude")

    # ---------------------------
    # B) Standard numeric handling
    # ---------------------------
    if req.latitude is not None and req.longitude is not None:
        try:
            lat = float(req.latitude)
            lon = float(req.longitude)
        except:
            raise HTTPException(status_code=400, detail="Latitude and Longitude must be numbers.")

        is_covered, details = checker.check_point(lat, lon)
        return CoverageResponse(
            address="Coordinates Only",
            latitude=lat,
            longitude=lon,
            in_coverage=is_covered,
            details=details
        )

    # ---------------------------
    # C) Address lookup
    # ---------------------------
    if req.address:
        if not geolocator:
            raise HTTPException(status_code=500, detail="Google Maps API Key missing.")

        location = geolocator.geocode(req.address)
        if not location:
            raise HTTPException(status_code=404, detail="Address not found.")

        is_covered, details = checker.check_point(location.latitude, location.longitude)
        return CoverageResponse(
            address=location.address,
            latitude=location.latitude,
            longitude=location.longitude,
            in_coverage=is_covered,
            details=details
        )

    raise HTTPException(status_code=400, detail="Send latitude+longitude OR address.")


# ==========================================
# GET /check-get (easy browser testing)
# ==========================================
@app.get("/check-get", response_model=CoverageResponse)
async def check_get(lat: float, lon: float):
    is_covered, details = checker.check_point(lat, lon)
    return CoverageResponse(
        address="Coordinates Only",
        latitude=lat,
        longitude=lon,
        in_coverage=is_covered,
        details=details
    )


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
