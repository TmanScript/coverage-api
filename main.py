import os
import zipfile
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.geocoders import GoogleV3
from bs4 import BeautifulSoup

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

                            # Point
                            point_tag = p.find('Point')
                            if point_tag:
                                coords_tag = point_tag.find('coordinates')
                                if coords_tag:
                                    coords = self.parse_coords_string(coords_tag.text)
                                    if coords:
                                        geometry = Point(coords[0])

                            # Polygon
                            poly_tag = p.find('Polygon')
                            if poly_tag:
                                outer_bound = poly_tag.find('outerBoundaryIs')
                                if outer_bound:
                                    coords_tag = outer_bound.find('coordinates')
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

        # Check Polygons
        if not self.polygons.empty:
            matches = self.polygons.contains(user_point)
            if matches.any():
                hit = self.polygons[matches].iloc[0]
                details = {k: v for k, v in hit.to_dict().items() if k != 'geometry'}
                details['match_type'] = 'Inside Polygon Coverage'
                return True, details

        # Check Towers
        if not self.points.empty:
            user_point_proj = gpd.GeoSeries([user_point], crs="EPSG:4326").to_crs("EPSG:3857")[0]
            points_proj = self.points.to_crs("EPSG:3857")
            distances = points_proj.distance(user_point_proj)

            radius_meters = COVERAGE_RADIUS_KM * 1000
            nearby_mask = distances <= radius_meters

            if nearby_mask.any():
                nearest_idx = distances.idxmin()
                nearest_tower = self.points.iloc[nearest_idx].to_dict()
                dist_val = distances.min()
                details = {k: v for k, v in nearest_tower.items() if k != 'geometry'}
                details['match_type'] = 'Tower Proximity'
                details['distance_km'] = round(dist_val / 1000, 2)
                return True, details

        return False, None

# ==========================================
# FASTAPI SETUP
# ==========================================
app = FastAPI(title="Coverage Check API (JSON Input)")

checker = None
try:
    checker = CoverageChecker(KMZ_FILE)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load KMZ. {e}")

# Google Maps geocoder
try:
    geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY, timeout=10)
except Exception as e:
    print("Warning: Google Maps API Key missing or invalid.")
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
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None  # optional address field


# ==========================================
# MAIN ENDPOINT (JSON POST)
# ==========================================
@app.post("/check", response_model=CoverageResponse)
async def check_coverage_json(req: CoordsRequest):
    if not checker:
        raise HTTPException(status_code=500, detail="Coverage map not loaded.")

    # ---- CASE 1: Latitude + Longitude given ----
    if req.latitude is not None and req.longitude is not None:
        is_covered, details = checker.check_point(req.latitude, req.longitude)
        return CoverageResponse(
            address="Coordinates Only",
            latitude=req.latitude,
            longitude=req.longitude,
            in_coverage=is_covered,
            details=details
        )

    # ---- CASE 2: Use address ----
    if req.address:
        if not geolocator:
            raise HTTPException(status_code=500, detail="Google Maps API Key not configured.")
        try:
            location = geolocator.geocode(req.address)
            if not location:
                raise HTTPException(status_code=404, detail="Address not found by Google Maps.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Google Maps Error: {str(e)}")

        is_covered, details = checker.check_point(location.latitude, location.longitude)

        return CoverageResponse(
            address=location.address,
            latitude=location.latitude,
            longitude=location.longitude,
            in_coverage=is_covered,
            details=details
        )

    # ---- CASE 3: Neither coordinates nor address ----
    raise HTTPException(status_code=400, detail="JSON must include either latitude+longitude OR address.")


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
