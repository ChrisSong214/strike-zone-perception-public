import os
import sys
from pybaseball import statcast
import pandas as pd
import numpy as np
import datetime
import urllib.request
from io import StringIO

class StatcastDataHandler:
    def __init__(self):
        self.type_conversion = {
            'game_pk': int,
            'pitch_type': str,
            'game_date': 'datetime64[ns]',
            'game_year': int,
            'release_speed': float,
            'release_pos_x': float,
            'release_pos_z': float,
            'player_name': str,
            'batter': int,
            'pitcher': int,
            'events': str,
            'description': str,
            'spin_dir': float,
            'zone': str,
            'des': str,
            'game_type': str,
            'stand': str,
            'p_throws': str,
            'home_team': str,
            'away_team': str,
            'type': str,
            'hit_location': float,
            'bb_type': str,
            'balls': int,
            'strikes': int,
            'pfx_x': float,
            'pfx_z': float,
            'plate_x': float,
            'plate_z': float,
            'on_3b': float,
            'on_2b': float,
            'on_1b': float,
            'outs_when_up': int,
            'inning': int,
            'inning_topbot': str,
            'fielder_2': int,
            'fielder_3': int,
            'fielder_4': int,
            'fielder_5': int,
            'fielder_6': int,
            'fielder_7': int,
            'fielder_8': int,
            'fielder_9': int,
            'umpire': float,
            'vx0': float,
            'vy0': float,
            'vz0': float,
            'ax': float,
            'ay': float,
            'az': float,
            'sz_top': float,
            'sz_bot': float,
            'hc_x': float,
            'hc_y': float,
            'hit_distance_sc': float,
            'launch_speed': float,
            'launch_angle': float,
            'effective_speed': float,
            'release_spin_rate': float,
            'spin_axis': float,
            'release_extension': float,
            'release_pos_y': float,
            'estimated_ba_using_speedangle': float,
            'estimated_woba_using_speedangle': float,
            'woba_value': float,
            'woba_denom': float,
            'babip_value': float,
            'iso_value': float,
            'launch_speed_angle': float,
            'at_bat_number': int,
            'pitch_number': int,
            'pitch_name': str,
            'home_score': int,
            'away_score': int,
            'bat_score': int,
            'fld_score': int,
            'post_away_score': int,
            'post_home_score': int,
            'post_bat_score': int,
            'post_fld_score': int,
            'if_fielding_alignment': str,
            'of_fielding_alignment': str,
            'delta_home_win_exp': float,
            'delta_run_exp': float,
            'bat_speed': float,
            'swing_length': float,
        }
        self.keep_cols = list(np.sort(list(self.type_conversion.keys())))
        
        # Robust path handling: always relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, '../data')
        self.savant_dir = os.path.join(self.data_dir, 'raw_data')

    def get_player_meta(self, update=False):
        path = os.path.join(self.data_dir, 'player_meta.feather')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path) and not update:
            print('Reading player metadata from file...')
            return pd.read_feather(path)
        else:
            print('Fetching new player metadata...')
            idx_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']
            dfs = []
            for i in idx_lst:
                url = f'https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{i}.csv'
                print(url)
                try:
                    with urllib.request.urlopen(url) as response:
                        csv_data = response.read().decode('utf-8')
                        df = pd.read_csv(StringIO(csv_data), low_memory=False)
                        dfs.append(df)
                except Exception as e:
                    print(f"Failed to fetch {url}: {e}")
            
            if not dfs:
                print("Warning: Could not fetch player metadata.")
                return pd.DataFrame()

            player_df = (
                pd.concat(dfs, ignore_index=True)
                .copy()
                .assign(name_full=lambda x: x['name_first'] + ' ' + x['name_last'])
            )[['key_person', 'key_uuid', 'key_mlbam', 'key_retro', 'key_bbref', 'key_fangraphs', 'name_first', 'name_last', 'name_full']]
            
            player_df = (
                player_df
                .dropna(subset=['key_mlbam'])
                .astype({'key_mlbam': int})
                .reset_index(drop=True)
            )
            
            player_df.to_feather(path)
            return player_df

    def update_local_sc(self, just_current=True, specific_year=None):
        player_df = self.get_player_meta(update=False)
        current_year = datetime.datetime.now().year
        
        if specific_year:
            years_list = [specific_year]
        elif just_current:
            years_list = [current_year]
            # If current year is 2026 and it's Jan, maybe fallback to 2025?
            # Or just let it try 2026.
        else:
            years_list = list(range(2015, current_year + 1))
        
        os.makedirs(self.savant_dir, exist_ok=True)

        for year in years_list:
            year_path = os.path.join(self.savant_dir, f'{year}.feather')
            
            # Default dates
            start_date = f'{year}-02-15'
            end_date = f'{year}-11-15'
            
            is_update = False
            sc_dat_saved = None

            if os.path.exists(year_path):
                print(f"Reading existing data for {year}...")
                sc_dat_saved = pd.read_feather(year_path)
                if not sc_dat_saved.empty and just_current and not specific_year:
                    # If we only want current/incremental, find last date
                    if 'game_date' in sc_dat_saved.columns:
                        last_saved = sc_dat_saved['game_date'].max()
                        if pd.notna(last_saved):
                            start_date = (last_saved + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                            is_update = True
                            
                            if pd.to_datetime(start_date) > datetime.datetime.now():
                                print(f"{year} data is up to date.")
                                continue

            print(f"Fetching {year} from {start_date} to {end_date}...")
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                 print("Start date after end date, skipping.")
                 continue

            try:
                sc_dat_tmp = statcast(start_dt=start_date, end_dt=end_date, verbose=True)
            except Exception as e:
                print(f"Error fetching data: {e}")
                continue

            if sc_dat_tmp is None or sc_dat_tmp.empty:
                print(f"No new data for {year}")
                if not is_update and not os.path.exists(year_path):
                     pass
                continue

            sc_dat_tmp = sc_dat_tmp.reset_index(drop=True)
            
            # Enforce types and columns
            cols_to_keep = [c for c in self.keep_cols if c in sc_dat_tmp.columns]
            sc_dat_tmp = sc_dat_tmp[cols_to_keep].copy()
            
            for col, dtype in self.type_conversion.items():
                if col in sc_dat_tmp.columns:
                    try:
                        sc_dat_tmp[col] = sc_dat_tmp[col].astype(dtype)
                    except:
                        pass # Squelch conversion errors

            # Merge player names
            if not player_df.empty:
                sc_dat_tmp = (
                    sc_dat_tmp
                    .merge(player_df[['key_mlbam', 'name_full']].rename(columns={'name_full': 'batter_name', 'key_mlbam': 'batter'}), on='batter', how='left')
                    .merge(player_df[['key_mlbam', 'name_full']].rename(columns={'name_full': 'pitcher_name', 'key_mlbam': 'pitcher'}), on='pitcher', how='left')
                )
            
            if is_update and sc_dat_saved is not None:
                sc_dat_tmp = pd.concat([sc_dat_saved, sc_dat_tmp], axis=0, ignore_index=True).reset_index(drop=True)
            
            print(f"Saving {len(sc_dat_tmp)} rows to {year_path}")
            sc_dat_tmp.to_feather(year_path)

    def fetch_statcast(self, start_year=None, end_year=None, game_types=['R', 'F', 'D', 'L', 'W']):
        """
        Loads data from local feather files.
        """
        if start_year is None: start_year = 2020
        if end_year is None: end_year = datetime.datetime.now().year

        df_list = []
        
        print(f'Loading data from {start_year} to {end_year}...')
        for year in range(int(start_year), int(end_year) + 1):
            year_path = os.path.join(self.savant_dir, f'{year}.feather')
            if os.path.exists(year_path):
                tmp_df = pd.read_feather(year_path)
                if 'game_type' in tmp_df.columns:
                    tmp_df = tmp_df[tmp_df['game_type'].isin(game_types)]
                df_list.append(tmp_df)
            else:
                print(f"Warning: Data for {year} not found at {year_path}.")

        if not df_list:
            return pd.DataFrame()

        df = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)
        return df

if __name__ == "__main__":
    handler = StatcastDataHandler()
    print("Initializing Data Handler...")
    
    # User requested to run for 2025 specifically
    print("Updating 2025 data...")
    handler.update_local_sc(just_current=False, specific_year=2025)
