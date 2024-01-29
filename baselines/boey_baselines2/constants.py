STATES_TO_SAVE_LOAD = ['recent_frames', 'agent_stats', 'base_explore', 'max_opponent_level', 'max_event_rew', 'max_level_rew', 
                       'last_health', 'last_num_poke', 'last_num_mon_in_box', 'total_healing_rew', 'died_count', 'prev_knn_rew', 
                       'visited_pokecenter_list', 'last_10_map_ids', 'last_10_coords', 'past_events_string', 'last_10_event_ids', 
                       'early_done', 'step_count', 'past_rewards', 'base_event_flags', 'rewarded_events_string', 'seen_map_dict', 
                       '_cut_badge', '_have_hm01', '_can_use_cut', '_surf_badge', '_have_hm03', '_can_use_surf', '_have_pokeflute', 
                       '_have_silph_scope', 'used_cut_coords_dict', '_last_item_count', '_is_box_mon_higher_level', 'hideout_elevator_maps', 
                       'use_mart_count', 'use_pc_swap_count',
                       'seen_coords', 'perm_seen_coords', 'special_seen_coords_count', 'secret_switch_states',
                       'party_level_base', 'party_level_post']

GYM_INFO = [
    {
        'badge': 0,
        'num_poke': 2,
        'max_level': 14,
    },
    {
        'badge': 1,
        'num_poke': 2,
        'max_level': 21,
    },
    {
        'badge': 2,
        'num_poke': 3,
        'max_level': 24,
    },
    {
        'badge': 3,
        'num_poke': 3,
        'max_level': 29,
    },
    {
        'badge': 4,
        'num_poke': 4,
        'max_level': 43,
    },
    {
        'badge': 5,
        'num_poke': 4,
        'max_level': 43,
    },
    {
        'badge': 6,
        'num_poke': 4,
        'max_level': 47,
    },
    {
        'badge': 7,
        'num_poke': 6,
        'max_level': 100,
    },
    # {
    #     'badge': 7,
    #     'num_poke': 5,
    #     'max_level': 50,
    # },
    # {
    #     'badge': 8,
    #     'num_poke': 5,
    #     'max_level': 56,
    # },
    # {
    #     'badge': 9,
    #     'num_poke': 5,
    #     'max_level': 58,
    # },
    # {
    #     'badge': 10,
    #     'num_poke': 5,
    #     'max_level': 60,
    # },
    # {
    #     'badge': 11,
    #     'num_poke': 5,
    #     'max_level': 62,
    # },
    {
        'badge': 12,
        'num_poke': 6,
        'max_level': 100,
    },
]

CAVE_MAP_IDS = [
    0x3B,  # Mt. Moon 1F
    0x3C,  # Mt. Moon B1F
    0x3D,  # Mt. Moon B2F
    0x52,  # Rock Tunnel 1F
    0xC5,  # DIGLETTS_CAVE
    0xE8,  # ROCK_TUNNEL_B1F
    0x6C,  # VICTORY_ROAD_1F
    0xC6,  # VICTORY_ROAD_3F
    0xC2,  # VICTORY_ROAD_2F
    0xE2,  # CERULEAN_CAVE_2F
    0xE4,  # CERULEAN_CAVE_1F
    0xE3,  # CERULEAN_CAVE_B1F
    0xC0,  # SEAFOAM_ISLANDS_1F
    0x9F,  # SEAFOAM_ISLANDS_B1F
    0xA0,  # SEAFOAM_ISLANDS_B2F
    0xA1,  # SEAFOAM_ISLANDS_B3F
    0xA2,  # SEAFOAM_ISLANDS_B4F
    0x2E,  # DIGLETTS_CAVE_ROUTE_2
    0x55,  # DIGLETTS_CAVE_ROUTE_11
]

GYM_MAP_IDS = [
    0x36,  # Pewter City Gym
    0x41,  # Cerulean City Gym
    0x5C,  # Vermilion City Gym
    0x86,  # Celadon City Gym
    0x9D,  # Fuchsia City Gym
    0xB2,  # Saffron City Gym
    0xA6,  # Cinnabar Island Gym
    0x2D,  # Viridian City Gym
    # Elite Four
    0xF5,  # Lorelei Room
    0xF6,  # Bruno Room
    0xF7,  # Agatha Room
    0x71,  # Lance Room
    0x78,  # Champion Room
]

ETC_MAP_IDS = [
    # SS Anne
    0x5F,  # 1F
    0x60,  # 2F
    0x61,  # 3F
    0x62,  # B1F
    0x63,  # SS_ANNE_BOW
    0x64,  # SS_ANNE_KITCHEN
    0x65,  # SS_ANNE_CAPTAINS_ROOM
    0x66,  # SS_ANNE_1F_ROOMS
    0x67,  # SS_ANNE_2F_ROOMS
    0x68,  # SS_ANNE_B1F_ROOMS
    # Power Plant
    0x53,  # Power Plant
    # Rocket Hideout
    0x87,  # GAME_CORNER
    0xC7,  # ROCKET_HIDEOUT_B1F
    0xC8,  # ROCKET_HIDEOUT_B2F
    0xC9,  # ROCKET_HIDEOUT_B3F
    0xCA,  # ROCKET_HIDEOUT_B4F
    0xCB,  # ROCKET_HIDEOUT_ELEVATOR
    # Silph Co
    0xB5,  # SILPH_CO_1F
    0xCF,  # SILPH_CO_2F
    0xD0,  # SILPH_CO_3F
    0xD1,  # SILPH_CO_4F
    0xD2,  # SILPH_CO_5F
    0xD3,  # SILPH_CO_6F
    0xD4,  # SILPH_CO_7F
    0xD5,  # SILPH_CO_8F
    0xE9,  # SILPH_CO_9F
    0xEA,  # SILPH_CO_10F
    0xEB,  # SILPH_CO_11F
    0xEC,  # SILPH_CO_ELEVATOR
    # Pokémon Tower
    0x8E,  # POKEMON_TOWER_1F
    0x8F,  # POKEMON_TOWER_2F
    0x90,  # POKEMON_TOWER_3F
    0x91,  # POKEMON_TOWER_4F
    0x92,  # POKEMON_TOWER_5F
    0x93,  # POKEMON_TOWER_6F
    0x94,  # POKEMON_TOWER_7F
    # Pokémon Mansion
    0xA5,  # POKEMON_MANSION_1F
    0xD6,  # POKEMON_MANSION_2F
    0xD7,  # POKEMON_MANSION_3F
    0xD8,  # POKEMON_MANSION_B1F
    # Safari Zone
    0x9C,  # SAFARI_ZONE_GATE
    0xD9,  # SAFARI_ZONE_EAST
    0xDA,  # SAFARI_ZONE_NORTH
    0xDB,  # SAFARI_ZONE_WEST
    0xDC,  # SAFARI_ZONE_CENTER
    0xDD,  # SAFARI_ZONE_CENTER_REST_HOUSE
    0xDE,  # SAFARI_ZONE_SECRET_HOUSE
    0xDF,  # SAFARI_ZONE_WEST_REST_HOUSE
    0xE0,  # SAFARI_ZONE_EAST_REST_HOUSE
    0xE1,  # SAFARI_ZONE_NORTH_REST_HOUSE
    # Sea routes
    0x1E,  # ROUTE_19
    0x1F,  # ROUTE_20
    0x20,  # ROUTE_21
]

SPECIAL_MAP_IDS = [] + CAVE_MAP_IDS + GYM_MAP_IDS + ETC_MAP_IDS


IGNORED_EVENT_IDS = [
    30,  # enter town map house
    29,  # leave town map house
    111,  # museum ticket
    1314,  # route 22 first rival battle
    1016,  # magikrap trade in Mt Moon Pokecenter
]


SPECIAL_KEY_ITEM_IDS = [
    0x30,  # CARD_KEY
    0x2B,  # SECRET_KEY
    0x48,  # SILPH_SCOPE
    0x4A,  # LIFT_KEY
    0x49,  # POKE_FLUTE
    0x3F,  # S_S_TICKET
    0x06,  # BICYCLE
    0x40,  # GOLD_TEETH
    0x3C,  # FRESH_WATER
    # 0x3D,  # SODA_POP
    # 0x3E,  # LEMONADE
]


ALL_KEY_ITEMS = [
    0x05,  # TOWN_MAP
    0x06,  # BICYCLE
    0x07,  # SURFBOARD
    0x08,  # SAFARI_BALL
    0x09,  # POKEDEX
    0x15,  # BOULDERBADGE
    0x16,  # CASCADEBADGE
    0x17,  # THUNDERBADGE
    0x18,  # RAINBOWBADGE
    0x19,  # SOULBADGE
    0x1A,  # MARSHBADGE
    0x1B,  # VOLCANOBADGE
    0x1C,  # EARTHBADGE
    0x20,  # OLD_AMBER
    0x29,  # DOME_FOSSIL
    0x2A,  # HELIX_FOSSIL
    0x2B,  # SECRET_KEY
    0x2D,  # BIKE_VOUCHER
    0x30,  # CARD_KEY
    0x3F,  # S_S_TICKET
    0x40,  # GOLD_TEETH
    0x45,  # COIN_CASE
    0x46,  # OAKS_PARCEL
    0x47,  # ITEMFINDER
    0x48,  # SILPH_SCOPE
    0x49,  # POKE_FLUTE
    0x4A,  # LIFT_KEY
    0x4C,  # OLD_ROD
    0x4D,  # GOOD_ROD
    0x4E,  # SUPER_ROD
    # quest items to keep
    0x3C,  # FRESH_WATER
    # 0x3D,  # SODA_POP
    # 0x3E,  # LEMONADE
]

ALL_HM_IDS = [
    0xc4,  # CUT
    0xc5,  # FLY
    0xc6,  # SURF
    0xc7,  # STRENGTH
    0xc8,  # FLASH
]

ALL_POKEBALL_IDS = [
    0x01,  # MASTER_BALL
    0x02,  # ULTRA_BALL
    0x03,  # GREAT_BALL
    0x04,  # POKE_BALL
]

	# const FULL_RESTORE  ; $10
	# const MAX_POTION    ; $11
	# const HYPER_POTION  ; $12
	# const SUPER_POTION  ; $13
	# const POTION        ; $14
	# const FULL_HEAL     ; $34
	# const REVIVE        ; $35
	# const MAX_REVIVE    ; $36
	# const ELIXER        ; $52
	# const MAX_ELIXER    ; $53
ALL_HEALABLE_ITEM_IDS = [  # from worst to best, so that it will consume the worst first
    0x14,  # POTION
    0x13,  # SUPER_POTION
    0x12,  # HYPER_POTION
    0x11,  # MAX_POTION
    0x35,  # REVIVE
    0x34,  # FULL_HEAL
    0x10,  # FULL_RESTORE
    0x36,  # MAX_REVIVE
    0x52,  # ELIXER
    0x53,  # MAX_ELIXER
]

ALL_GOOD_ITEMS = ALL_KEY_ITEMS + ALL_POKEBALL_IDS + ALL_HEALABLE_ITEM_IDS + ALL_HM_IDS

GOOD_ITEMS_PRIORITY = [  # from worst to best, so that it will toss the worst first
    0x04,  # POKE_BALL
    0x14,  # POTION
    0x03,  # GREAT_BALL
    0x13,  # SUPER_POTION
    0x12,  # HYPER_POTION
    0x02,  # ULTRA_BALL
    0x35,  # REVIVE
    0x50,  # ETHER
    0x51,  # MAX_ETHER
    0x01,  # MASTER_BALL
    0x11,  # MAX_POTION
    0x34,  # FULL_HEAL
    0x52,  # ELIXER
    0x53,  # MAX_ELIXER
    0x10,  # FULL_RESTORE
    0x36,  # MAX_REVIVE
]

POKEBALL_PRIORITY = [
    # 0x01,  # MASTER_BALL  # not purchasable
    0x02,  # ULTRA_BALL
    0x03,  # GREAT_BALL
    0x04,  # POKE_BALL
]

POTION_PRIORITY = [
    0x10,  # FULL_RESTORE
    0x11,  # MAX_POTION
    0x12,  # HYPER_POTION
    0x13,  # SUPER_POTION
    0x14,  # POTION
]

REVIVE_PRIORITY = [
    0x36,  # MAX_REVIVE
    0x35,  # REVIVE
]

LEVELS = [
    {  # 0 level 1
        'last_pokecenter': ['VERMILION_CITY', 'LAVENDER_TOWN']
    },
    {  # 1 level 2
        'last_pokecenter': ['CELADON_CITY'],
    },
    {  # 2 level 3
        'last_pokecenter': ['LAVENDER_TOWN'],
    },
    {  # 3 level 4
        'badge': 5,
    },
    {  # 4 level 5
        'last_pokecenter': ['FUCHSIA_CITY'],
    },
    {  # 5 level 6
        'last_pokecenter': ['CINNABAR_ISLAND'],
    },
    {  # 6 level 7
        'last_pokecenter': ['VIRIDIAN_CITY'],
    },
    {  # 7 level 8
        'event': 'CHAMPION',
    },
]