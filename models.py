from PIL import Image
from e2p import Equirectangular
import json
import asyncio, aiohttp
import io
import numpy as np
import os
import sqlite3
from math import radians, sin, cos, sqrt, atan2
from typing import Self
import logging
from config import MAPS

GSV_PANO_URL = "https://geo0.ggpht.com/cbk"

class Pano:
    """
    A GSV panorama, with a unique ID and image file.
    """
    def __init__(self, pano_id=None, lat=None, lng=None):
        self.zoom = 4
        self.dimensions = None
        self.driving_direction = None

        if lat is not None and lng is not None:
            self.pano_id = None
            self.lat = lat
            self.lng = lng
        else:
            self.pano_id = self.convert_pano_id(pano_id)
            self.lat = None
            self.lng = None
            
        self.panorama = None
        self.img = None
    
    
    async def get_panorama(self, heading, pitch, FOV=110):
        if self.pano_id is None:
            self.pano_id = await self.get_panoid()

        if self.panorama is None:
            await self.get_pano_metadata()
            self.panorama = await self._fetch_and_build_panorama()

        equ = Equirectangular(self.panorama)
        result = equ.GetPerspective(FOV, (heading - self.driving_direction) % 360, pitch, 1080, 1920)

        return result

    async def fetch_single_tile(self, session, x, y, retries=3):
        params = {
            "cb_client": "apiv3",
            "panoid": self.pano_id,
            "output": "tile",
            "zoom": self.zoom,
            "x": x,
            "y": y
        }
        
        for attempt in range(retries):
            try:
                async with session.get(GSV_PANO_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logging.error(f"Error fetching tile {x},{y}: Status {response.status}")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                    data = await response.read()
                    tile = Image.open(io.BytesIO(data))
                    return tile
            except Exception as e:
                logging.error(f"Exception fetching tile {x},{y}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None

    async def _fetch_and_build_panorama(self):        
        if self.dimensions[1] == 8192:  # Gen 4
            max_x, max_y = 16, 8
        elif self.dimensions[1] == 6656:  # Gen 3
            max_x, max_y = 13, 6.5
        else:  # Fallback
            max_x, max_y = 7, 4
        
        async with aiohttp.ClientSession() as session:
            # Get tiles based on determined dimensions
            raw_tiles = await asyncio.gather(
                *[self.fetch_single_tile(session, x, y) 
                for y in range(int(max_y) if max_y == 8 else 7)  # Handle Gen 3's 6.5
                for x in range(max_x)]
            )
            
            if max_y == 6.5:  # Handle Gen 3's half row
                tiles = []
                # First 6 rows
                for y in range(6):
                    for x in range(13):
                        idx = y * max_x + x
                        tiles.append(raw_tiles[idx])
                        
                # Half of 7th row
                for x in range(13):
                    idx = 6 * max_x + x
                    tile = raw_tiles[idx]
                    if tile is None:
                        continue
                    tile_array = np.array(tile)
                    half_height = tile_array.shape[0] // 2
                    half_tile = Image.fromarray(tile_array[:half_height])
                    tiles.append(half_tile)
                
                return self._stitch_panorama(tiles, max_x, max_y)
            else:
                return self._stitch_panorama(raw_tiles, max_x, max_y)
    
    def _stitch_panorama(self, tiles, max_x, max_y):
        is_half_height = max_y % 1 != 0
        full_height = int(max_y)
        
        tile_width, tile_height = tiles[0].size
        if is_half_height:
            last_row_height = tile_height // 2
            total_height = (full_height * tile_height) + last_row_height
        else:
            total_height = int(max_y * tile_height)
            
        total_width = int(max_x * tile_width)
        
        full_panorama = Image.new('RGB', (total_width, total_height))
        
        for idx, img in enumerate(tiles):
            x = (idx % int(max_x)) * tile_width
            y = (idx // int(max_x)) * tile_height
            full_panorama.paste(img, (x, y))
            
        return np.array(full_panorama)

    async def get_panoid(self):
        url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch"

        headers = {
            "Content-Type": "application/json+protobuf"
        }
        radius = 50
        payload = f'[["apiv3"],[[null,null,{self.lat},{self.lng}],{radius}],[[null,null,null,null,null,null,null,null,null,null,[null,null]],null,null,null,null,null,null,null,[1],null,[[[2,true,2]]]],[[2,6]]]'

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload, headers=headers) as response:
                try:
                    data = await response.json()
                    return data[1][1][1]
                except Exception as e:
                    logging.error(f"Error getting panoid: {e}")
    
    async def get_pano_metadata(self):
        if self.dimensions and self.driving_direction:
            return self.dimensions, self.driving_direction
        
        url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata"
    
        headers = {
            "Content-Type": "application/json+protobuf"
        }
        
        request_data = [
            ["apiv3",None,None,None,"US",None,None,None,None,None,[[0]]],
            ["en","US"],
            [[[2,self.pano_id]]],
            [[1,2,3,4,8,6]]
        ]

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data, headers=headers) as response:
                try:
                    data = await response.json()

                    self.dimensions = data[1][0][2][3][0][4][0]  # [height, width]
                    self.driving_direction = data[1][0][5][0][1][2][0]  # Driving direction
                    logging.info(f"Metadata: {self.dimensions}, {self.driving_direction}")
                    
                    return self.dimensions, self.driving_direction
                except Exception as e:
                    logging.error(f"Error getting metadata: {e}")
                    return None

    @staticmethod
    def convert_pano_id(pano_id):
        """
        Convert GeoGuessr hex-encoded pano ID to base64-encoded pano ID.
        """
        try:
            decoded = bytes.fromhex(pano_id).decode()
            return decoded
        except ValueError:
            return pano_id
        except Exception as e:
            logging.error(f"Error: {e}")
            return pano_id

    @staticmethod
    def add_compass(image: np.ndarray, heading: float, output_path: str = 'image.jpg'):
        """
        Add a compass overlay to an image.
        
        Args:
            image: numpy array of the image
            heading: Heading angle in degrees (0-360)
            output_path: Path to save the resulting image
        """
        try:
            # Convert numpy array to PIL Image
            main_image = Image.fromarray(image)
            compass = Image.open('compass.png')
            
            compass_size = int(min(main_image.size) * 0.15)  # 15% of smaller dimension
            compass = compass.resize((compass_size, compass_size), Image.Resampling.LANCZOS)
            
            compass = compass.convert('RGBA')
            rotated_compass = compass.rotate(heading, expand=False, resample=Image.Resampling.BICUBIC)
            
            # Calculate position (bottom left with padding)
            padding = int(compass_size * 0.2)  # 20% of compass size as padding
            position = (padding, main_image.size[1] - compass_size - padding)
            
            result = main_image.convert('RGBA')
            result.paste(rotated_compass, position, rotated_compass)
            
            result = result.convert('RGB')
            return result
            
        except Exception as e:
            logging.error(f"Error adding compass overlay: {e}")
    
    def to_dict(self):
        """Return a JSON-serializable representation of the Pano"""
        return {
            'pano_id': self.pano_id,
            'zoom': self.zoom,
            'dimensions': self.dimensions,
            'driving_direction': self.driving_direction
        }

class Round:
    """
    A round in a GeoGuessr game.

    Attributes:
        pano (Pano): Panorama object
        heading (float): Camera heading
        pitch (float): Camera pitch
        zoom (float): Camera zoom
        lat (float): Latitude
        lng (float): Longitude
        subdivision (str): Subdivision name
        locality (str): Locality name
    """
    def __init__(self, round_data):
        self.pano = Pano(pano_id=round_data['panoId']) if round_data['panoId'] else Pano(lat=round_data['lat'], lng=round_data['lng'])
        self.heading = round_data['heading']
        self.pitch = round_data['pitch']
        self.zoom = round_data['zoom']
        self.lat = round_data['lat']
        self.lng = round_data['lng']
        self.subdivision = None
        self.locality = None

    @staticmethod
    async def get_location_info(lat: float, lng: float):
        """Get location info from BigDataCloud reverse geocoding API
        
        Args:
            lat (float): Latitude
            lng (float): Longitude

        Returns:
            tuple:
                subdivision (str): Subdivision name
                locality (str): Locality name
        """
        url = "https://api.bigdatacloud.net/data/reverse-geocode"
        params = {
            "latitude": lat,
            "longitude": lng,
            "key": os.environ.get("BIGDATACLOUD_API_KEY")
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                logging.debug(f"Reverse geocoding status: {response.status}")
                if response.status == 403:
                    logging.error("API key not set or not authorized")
                    exit(1)
                if response.status != 200:
                    logging.error("Geocoding error")
                    return None
                    
                data = await response.json()
                
                return data.get('principalSubdivision'), data.get('locality')

    async def set_subdivision(self, round_data):            
        self.subdivision, self.locality = await self.get_location_info(
            round_data.get('lat'),
            round_data.get('lng')
        )
        return self.subdivision, self.locality
        
    @staticmethod
    async def reconstruct_round(round_data: dict, pano_processor) -> Self:
        """Helper to reconstruct a Round object from saved data"""
        pano_data = round_data['pano']
        round_obj = Round({
            'panoId': pano_data['pano_id'],
            'heading': round_data['heading'],
            'pitch': round_data['pitch'],
            'zoom': round_data['zoom'],
            'lat': round_data['lat'],
            'lng': round_data['lng']
        })
        round_obj.subdivision = round_data['subdivision']
        round_obj.locality = round_data['locality'] if 'locality' in round_data else None
        
        await pano_processor.process_pano(
            round_obj.pano,
            round_obj.heading,
            round_obj.pitch
        )
        return round_obj

    def to_dict(self):
        """Return a JSON-serializable representation of the Round"""
        return {
            'pano': self.pano.to_dict(),
            'heading': self.heading,
            'pitch': self.pitch,
            'zoom': self.zoom,
            'lat': self.lat,
            'lng': self.lng,
            'subdivision': self.subdivision,
            'locality': self.locality
        }

    @property
    def link(self):
        return f"https://www.google.com/maps/@{self.lat},{self.lng},3a,90y,{self.heading}h,{90-self.pitch}t/data=!3m7!1e1!3m5!1s{self.pano.pano_id}!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fpanoid%3D{self.pano.pano_id}%26!7i13312!8i6656"

class GameManager:
    """
    A game manager that handles state of individual channels.

    Attributes:
        db_path (str): Path to the SQLite database file
        subdivisions (list): List of subdivisions to use for location info
        rounds (dict): Current round data for each channel
        next_rounds (dict): Next round data for each channel
        waiting_for_guess (dict): Whether a channel is waiting for a guess
        streak (dict): Current streak count for each channel
        five_k_attempts (dict): 5k attempts for each user in each channel
    """
    def __init__(self, subdivisions, db_path="game_state.db"):
        self.db_path = db_path
        self.subdivisions = subdivisions
        self._init_db()
        self.rounds = {}
        self.next_rounds = {}
        self.waiting_for_guess = {}
        
        self.streak = {}
        self.five_k_attempts = {}
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Existing game state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS game_state (
                    channel_id INTEGER PRIMARY KEY,
                    streak INTEGER,
                    game_data TEXT,
                    current_round TEXT,
                    next_round TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS maps (
                    map_id TEXT PRIMARY KEY,
                    map_name TEXT
                )
            """)

            for map_id, map_info in MAPS.items():
                map_name = map_info[0]  # First element is the full name
                conn.execute("""
                    INSERT OR IGNORE INTO maps (map_id, map_name)
                    VALUES (?, ?)
                """, (map_id, map_name))
            
            # New rounds history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER,
                    user_id INTEGER,
                    streak_id INTEGER,
                    pano_id TEXT,
                    actual_location TEXT,
                    guessed_location TEXT,
                    is_correct BOOLEAN,
                    lat REAL,
                    lng REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    map TEXT,
                    FOREIGN KEY(channel_id) REFERENCES game_state(channel_id)
                    FOREIGN KEY(streak_id) REFERENCES streaks(id)
                    FOREIGN KEY(map) REFERENCES maps(map_id)
                )
            """)

            # Create some useful views for leaderboards
            conn.execute("""
                CREATE VIEW IF NOT EXISTS player_stats AS
                WITH streak_counts AS (
                    -- Count participants for each streak
                    SELECT 
                        streak_id,
                        COUNT(*) as participant_count
                    FROM streak_participants
                    GROUP BY streak_id
                ),
                streak_stats AS (
                    -- Get highest solo and assisted streaks for each player
                    SELECT 
                        sp.user_id,
                        MAX(CASE WHEN sc.participant_count = 1 THEN s.number ELSE 0 END) as best_solo_streak,
                        MAX(CASE WHEN sc.participant_count > 1 THEN s.number ELSE 0 END) as best_assisted_streak,
                        AVG(CASE WHEN sc.participant_count = 1 THEN s.number ELSE 0 END) as avg_solo_streak
                    FROM streak_participants sp
                    JOIN streaks s ON sp.streak_id = s.id
                    JOIN streak_counts sc ON s.id = sc.streak_id
                    GROUP BY sp.user_id
                )
                SELECT 
                    r.user_id,
                    COUNT(*) as total_guesses,
                    SUM(CASE WHEN r.is_correct THEN 1 ELSE 0 END) as correct_guesses,
                    ROUND(AVG(CASE WHEN r.is_correct THEN 100.0 ELSE 0 END), 1) as accuracy,
                    COALESCE(ss.best_solo_streak, 0) as best_solo_streak,
                    COALESCE(ss.best_assisted_streak, 0) as best_assisted_streak,
                    ROUND(COALESCE(ss.avg_solo_streak, 0), 2) as avg_solo_streak
                FROM rounds r
                LEFT JOIN streak_stats ss ON r.user_id = ss.user_id
                GROUP BY r.user_id;
            """)
            
            conn.execute("""
                CREATE VIEW IF NOT EXISTS player_subdivision_stats AS
                WITH player_location_stats AS (
                    SELECT 
                        user_id,
                        actual_location,
                        COUNT(*) as times_seen,
                        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as times_correct,
                        ROUND(AVG(CASE WHEN is_correct THEN 100.0 ELSE 0 END), 1) as accuracy_rate,
                        MAX(timestamp) as last_seen
                    FROM rounds
                    GROUP BY user_id, actual_location
                )
                SELECT 
                    user_id,
                    actual_location,
                    times_seen,
                    times_correct,
                    accuracy_rate,
                    last_seen,
                    RANK() OVER (
                        PARTITION BY user_id 
                        ORDER BY accuracy_rate ASC, times_seen DESC
                    ) as hardest_rank,
                    RANK() OVER (
                        PARTITION BY user_id 
                        ORDER BY accuracy_rate DESC, times_seen DESC
                    ) as easiest_rank
                FROM player_location_stats
                WHERE times_seen >= 3
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS streaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER,
                    number INTEGER NOT NULL,
                    start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_timestamp DATETIME,
                    FOREIGN KEY(channel_id) REFERENCES game_state(channel_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS streak_participants (
                    streak_id INTEGER,
                    user_id INTEGER,
                    guesses_count INTEGER DEFAULT 1,
                    PRIMARY KEY (streak_id, user_id),
                    FOREIGN KEY(streak_id) REFERENCES streaks(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS five_k_guesses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id INTEGER NOT NULL,
                    FOREIGN KEY (round_id) REFERENCES rounds(id)
                );
            """)
                    
    
    def log_round(self, channel_id: int, user_id: int, round_obj: Round, guess: str, actual: str, is_correct: bool, map_id: str):
        """Updates the database with data of a completed round.
        
        Args:
            channel_id (int): Channel ID
            user_id (int): User ID
            round_obj (Round): Round object
            guess (str): Guessed location (user)
            actual (str): Actual location (game)
            is_correct (bool): Whether the guess was correct
        """
        with sqlite3.connect(self.db_path) as conn:
            streak_id = conn.execute("""
                SELECT id FROM streaks 
                WHERE channel_id = ? AND end_timestamp IS NULL
                ORDER BY id DESC LIMIT 1
            """, (channel_id,)).fetchone()
            
            if is_correct:
                streak_record = conn.execute("""
                    SELECT id FROM streaks 
                    WHERE channel_id = ? AND end_timestamp IS NULL
                    ORDER BY id DESC LIMIT 1
                """, (channel_id,)).fetchone()
                
                if streak_record:
                    streak_id = streak_record[0]
                    conn.execute("""
                        UPDATE streaks SET number = ?
                        WHERE id = ?
                    """, (self.streak.get(channel_id), streak_id,))
                else:
                    cursor = conn.execute("""
                        INSERT INTO streaks (channel_id, number)
                        VALUES (?, ?)
                    """, (channel_id, self.streak.get(channel_id, 1),))
                    streak_id = cursor.lastrowid
                
                if streak_id:
                    conn.execute("""
                        INSERT INTO streak_participants (streak_id, user_id)
                        VALUES (?, ?)
                        ON CONFLICT(streak_id, user_id) DO UPDATE SET
                        guesses_count = guesses_count + 1
                    """, (streak_id, user_id))
            else:
                streak_record = conn.execute("""
                    SELECT id FROM streaks 
                    WHERE channel_id = ? AND end_timestamp IS NULL
                    ORDER BY id DESC LIMIT 1
                """, (channel_id,)).fetchone()
                
                if streak_record:
                    streak_id = streak_record[0]
                    conn.execute("""
                        UPDATE streaks SET end_timestamp = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (streak_id,))
                streak_id = None

            cur = conn.execute("""
                INSERT INTO rounds (
                    channel_id, user_id, streak_id, pano_id, 
                    actual_location, guessed_location, is_correct, 
                    lat, lng, map
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                channel_id,
                user_id,
                streak_id,
                round_obj.pano.pano_id,
                actual,
                guess,
                is_correct,
                round_obj.lat,
                round_obj.lng,
                map_id
            ))
            return cur.lastrowid

    def save_state(self, channel_id: int, game_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            current_round = json.dumps(self.rounds[channel_id].to_dict()) if channel_id in self.rounds else None
            next_round = json.dumps(self.next_rounds[channel_id].to_dict()) if channel_id in self.next_rounds else None
            
            conn.execute("""
                INSERT OR REPLACE INTO game_state 
                VALUES (?, ?, ?, ?, ?)
            """, (
                channel_id,
                self.streak.get(channel_id, 0),
                json.dumps(game_data),
                current_round,
                next_round
            ))

    def has_saved_state(self, channel_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM game_state WHERE channel_id = ?",
                (channel_id,)
            ).fetchone()
            return row is not None
    
    def load_state(self, channel_id: int) -> dict:
        """Load saved game state for a channel"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT streak, game_data, current_round, next_round FROM game_state WHERE channel_id = ?",
                (channel_id,)
            ).fetchone()
            
            if not row:
                return None
                
            return {
                'streak': row[0],
                'game_data': json.loads(row[1]) if row[1] else None,
                'current_round': json.loads(row[2]) if row[2] else None,
                'next_round': json.loads(row[3]) if row[3] else None
            }
    
    def end_streak(self, channel_id: int):
        """Force-end the current streak for a channel"""
        with sqlite3.connect(self.db_path) as conn:
            # End any active streak
            streak_record = conn.execute("""
                SELECT id FROM streaks 
                WHERE channel_id = ? AND end_timestamp IS NULL
                ORDER BY id DESC LIMIT 1
            """, (channel_id,)).fetchone()
            
            if streak_record:
                streak_id = streak_record[0]
                conn.execute("""
                    UPDATE streaks SET end_timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (streak_id,))
                
            # Reset streak counter
            self.streak[channel_id] = 0
    
    async def check_if_top_streak(self, channel_id: int, streak_number: int) -> list:
        """
        Check if a streak is in any of the top 5 leaderboards.
        Returns a list of tuples (category, position) for leaderboards where the streak places.
        """
        achievements = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Get the latest ended streak's participant info
            participants = conn.execute("""
                WITH latest_streak AS (
                    SELECT id, number 
                    FROM streaks 
                    WHERE channel_id = ? AND end_timestamp IS NULL
                    LIMIT 1
                )
                SELECT COUNT(DISTINCT user_id) as participant_count
                FROM streak_participants
                WHERE streak_id = (SELECT id FROM latest_streak)
            """, (channel_id,)).fetchone()
            
            if not participants:
                return achievements
                
            participant_count = participants[0]
            is_solo = participant_count == 1
            
            # Check all streaks category first
            base_query = """
                WITH streak_counts AS (
                    SELECT streak_id, COUNT(user_id) as participant_count
                    FROM streak_participants
                    GROUP BY streak_id
                ),
                ranked_streaks AS (
                    SELECT 
                        s.number,
                        RANK() OVER (ORDER BY s.number DESC) as position
                    FROM streaks s
                    JOIN streak_counts sc ON s.id = sc.streak_id
                    WHERE s.number > 0 {filter}
                )
                SELECT number, position
                FROM ranked_streaks
                WHERE position <= 5
                ORDER BY position
                """
            
            # Check "all" category
            all_top_streaks = conn.execute(base_query.format(filter="")).fetchall()
            if not all_top_streaks or streak_number >= min(s[0] for s in all_top_streaks):
                position = 1
                for rank, (number, _) in enumerate(all_top_streaks, 1):
                    if streak_number >= number:
                        position = rank
                        break
                    position = rank + 1
                if position <= 5:
                    achievements.append(("all", position))
            
            # Only check the specific category matching this streak's type
            category_filter = "AND sc.participant_count = 1" if is_solo else "AND sc.participant_count > 1"
            category_name = "solo" if is_solo else "assisted"
            
            category_top_streaks = conn.execute(base_query.format(filter=category_filter)).fetchall()
            if not category_top_streaks or streak_number >= min(s[0] for s in category_top_streaks):
                position = 1
                for rank, (number, _) in enumerate(category_top_streaks, 1):
                    if streak_number >= number:
                        position = rank
                        break
                    position = rank + 1
                if position <= 5:
                    achievements.append((category_name, position))
        
        return achievements
    

    def check_5k_guess(self, text: str) -> tuple[float, float] | None:
        """
        Check if text is a valid coordinate guess.
        Returns (lat, lng) tuple if valid, None otherwise.
        """
        text = text.lower().replace('!g', '').strip()
        
        try:
            parts = [p.strip() for p in text.split(',')]
            if len(parts) != 2:
                return None
                
            lat, lng = map(float, parts)
            
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return None
                
            return (lat, lng)
        except (ValueError, IndexError):
            return None

    def verify_5k_guess(self, channel_id: int, user_id: int, guess_coords: tuple[float, float], round_obj: Round) -> tuple[bool, float]:
        """
        Verify a 5K guess.
        Returns (is_correct, distance) tuple. First bool indicates if we can proceed (have attempts),
        distance is actual distance if we can proceed.
        """
        if not self.check_5k_attempts(channel_id, user_id):
            return (False, 0)
            
        distance = self.calculate_distance_meters(
            guess_coords[0], guess_coords[1],
            round_obj.lat, round_obj.lng
        )
        
        self.increment_5k_attempts(channel_id, user_id)
        
        return (True, distance)

    def reset_5k_attempts(self, channel_id):
        """Reset 5k attempts for a channel"""
        self.five_k_attempts[channel_id] = {}
        
    def check_5k_attempts(self, channel_id, user_id) -> bool:
        """Check if user has attempts remaining"""
        attempts = self.five_k_attempts.get(channel_id, {}).get(user_id, 0)
        return attempts < 5
        
    def increment_5k_attempts(self, channel_id, user_id):
        """Increment 5k attempts for a user"""
        if channel_id not in self.five_k_attempts:
            self.five_k_attempts[channel_id] = {}
        self.five_k_attempts[channel_id][user_id] = self.five_k_attempts[channel_id].get(user_id, 0) + 1

    @staticmethod
    def calculate_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
        