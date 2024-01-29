from .pokered_constants import MAP_DICT

STAGE_DICT = [
    {  # stage 0, index 0
        'trigger': [  # have ss anne ticket
            'have', 0x3F  # ss anne ticket
            ],
        'start': [
            ['block', 'CERULEAN_CITY', 'north',],
            ['block', 'CERULEAN_CITY', 'CERULEAN_TRASHED_HOUSE@1',],
        ],
        'clear': [
            ['have', 0x3F],  # ss anne ticket
            ['badge', 2],
        ],
        'end': [
            ['unblock', 'CERULEAN_CITY', 'CERULEAN_TRASHED_HOUSE@1',],
            ['block', 'VERMILION_CITY', 'north',],
            ['block', 'ROUTE_11', 'DIGLETTS_CAVE_ROUTE_11@1',],
        ]
    },
    {  # stage 1, index 1
        'trigger': [
            'pokecenter', 'VERMILION_CITY',
        ],
        'start': [
        ],
        'clear': [
            ['badge', 3],
            ['have', 0x2D],  # have BIKE_VOUCHER
        ],
        'end': [
            ['unblock', 'VERMILION_CITY', 'north',],
            ['block', 'ROUTE_6', 'south',],
            ['block', 'ROUTE_5', 'UNDERGROUND_PATH_ROUTE_5@1',],
        ]
    },
    {  # stage 1.5, index 2
        'trigger': [
            'enter', 'CERULEAN_CITY',
        ],
        'start': [
            ['block', 'CERULEAN_CITY', 'east',],
            # ['block', 'CERULEAN_CITY', 'south',],
        ],
        'clear': [
            ['have', 0x06],  # have BICYCLE
        ],
        'end': [
            ['unblock', 'CERULEAN_CITY', 'east',],
            ['block', 'ROUTE_10', 'ROCK_TUNNEL_1F@1',],
        ]
    },
    {  # stage 2, index 3
        'trigger': [
            'pokecenter', 'ROUTE_10',
        ],
        'start': [
            ['block', 'ROUTE_10', 'west',],
            ['unblock', 'ROUTE_10', 'ROCK_TUNNEL_1F@1',],
            ['block', 'LAVENDER_TOWN', 'west', ],
            ['block', 'LAVENDER_TOWN', 'POKEMON_TOWER_1F@1'],
            ['block', 'LAVENDER_TOWN', 'north', ],
            ['block', 'LAVENDER_TOWN', 'south'],
        ],
        'clear': [
            ['pokecenter', 'LAVENDER_TOWN'],
        ],
        'end': [
            ['unblock', 'ROUTE_10', 'west',],
            ['unblock', 'ROUTE_5', 'UNDERGROUND_PATH_ROUTE_5@1',],
            ['unblock', 'ROUTE_6', 'south',],
        ]
    },
    {  # stage 3, index 4
        'trigger': [
            'pokecenter', 'LAVENDER_TOWN',
        ],
        'start': [
            ['unblock', 'LAVENDER_TOWN', 'west', ],
            ['block', 'LAVENDER_TOWN', 'POKEMON_TOWER_1F@1'],
            ['block', 'CELADON_CITY', 'east', ],
        ],
        'clear': [
            ['pokecenter', 'CELADON_CITY'],
        ],
        # 'end': [
        #     'unblock', 'ROUTE_10', ['west',],
        # ]
    },
    {  # stage 4, index 5
        'trigger': [
            'pokecenter', 'CELADON_CITY',
        ],
        'start': [
        ],
        'clear': [
            ['have', 0x48],  # have SILPH_SCOPE
            ['badge', 4],
            ['have', 0x3C],  # FRESH_WATER only
        ],
        'end': [
            ['unblock', 'CELADON_CITY', 'east', ],
            ['block', 'SAFFRON_CITY', 'north', ],
            ['block', 'SAFFRON_CITY', 'south', ],
            ['block', 'ROUTE_8', 'UNDERGROUND_PATH_ROUTE_8@1', ],
            ['block', 'ROUTE_8', 'ROUTE_8_GATE@3', ],
            ['block', 'ROUTE_8', 'ROUTE_8_GATE@4', ],
            ['block', 'ROUTE_7', 'UNDERGROUND_PATH_ROUTE_7@1'],  # force it to go through saffron city
        ]
    },
    {  # stage 5, index 6
        'trigger': [
            # 'pokecenter', 'LAVENDER_TOWN',
            'last_pokecenter', 'LAVENDER_TOWN',
        ],
        'start': [
            ['unblock', 'LAVENDER_TOWN', 'POKEMON_TOWER_1F@1'],
            # ['block', 'LAVENDER_TOWN', 'north', ],
            # ['block', 'LAVENDER_TOWN', 'south', ],
        ],
        'clear': [
            ['have', 0x49],  # have POKEFLUTE
        ],
        'end': [
            ['unblock', 'ROUTE_8', 'ROUTE_8_GATE@3', ],
            ['unblock', 'ROUTE_8', 'ROUTE_8_GATE@4', ],
            ['block', 'SAFFRON_CITY', 'east', ],
            ['block', 'SAFFRON_CITY', 'west', ],
            ['block', 'LAVENDER_TOWN', 'POKEMON_TOWER_1F@1'],  # block it to ease progression
        ]
    },
    {  # stage 6, index 7
        'trigger': [
            'enter', 'SAFFRON_CITY',
        ],
        'start': [
        ],
        'clear': [
            ['event', 'EVENT_GOT_MASTER_BALL'],
        ],
        'end': [
            ['block', 'SAFFRON_CITY', 'SILPH_CO_1F@1', ],
        ],
        'events': ['EVENT_GOT_MASTER_BALL', ],
    },
    {  # stage 6.1, index 8
        # 'trigger': [
        #     'enter', 'SAFFRON_CITY',
        # ],
        'start': [
        ],
        'clear': [
            ['badge', 5],
        ],
        'end': [
            ['unblock', 'SAFFRON_CITY', 'east', ],
            ['unblock', 'SAFFRON_CITY', 'west', ],
            ['unblock', 'LAVENDER_TOWN', 'south'],
            ['block', 'ROUTE_12', 'west'],
            ['unblock', 'ROUTE_8', 'UNDERGROUND_PATH_ROUTE_8@1', ],  # no point keeps on blocking it
        ]
    },
    {  # stage 7, index 9
        'trigger': [
            'enter', 'FUCHSIA_CITY',
        ],
        'start': [
            ['block', 'FUCHSIA_CITY', 'east'],
            ['block', 'FUCHSIA_CITY', 'west'],
            ['block', 'ROUTE_19', 'west'],
        ],
        'clear': [
            ['badge', 6],
            ['have', 0xc7],  # have hm04
            ['event', 'CAN_USE_SURF'],
        ],
        'end': [
            ['block', 'CINNABAR_ISLAND', 'east'],
            ['block', 'CINNABAR_ISLAND', 'north'],
            ['unblock', 'ROUTE_19', 'west'],
        ],
        'events': ['CAN_USE_SURF', ],
    },
    {  # stage 8, index 10
        'trigger': [
            'pokecenter', 'CINNABAR_ISLAND',
        ],
        'start': [
        ],
        'clear': [
            ['badge', 7],
        ],
        'end': [
            ['unblock', 'CINNABAR_ISLAND', 'north'],
            ['block', 'VIRIDIAN_CITY', 'north'],
        ]
    },
    {  # stage 9, index 11
        'trigger': [
            'enter', 'VIRIDIAN_CITY',
        ],
        'start': [
            # ['block', 'VIRIDIAN_CITY', 'south'],  # let it roam back cinnabar freely
        ],
        'clear': [
            ['badge', 20],  # do not let it clear as this is the last stage
        ],
        # 'end': [
        # ]
    },
]

# pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A]
POKECENTER_TO_INDEX_DICT = {
    'VIRIDIAN_CITY': 0,
    'PEWTER_CITY': 1,
    'CERULEAN_CITY': 2,
    'ROUTE_4': 3,
    'ROUTE_10': 4,
    'VERMILION_CITY': 5,
    'CELADON_CITY': 6,
    'LAVENDER_TOWN': 7,
    'FUCHSIA_CITY': 8,
    'CINNABAR_ISLAND': 9,
    'SAFFRON_CITY': 10,
    'INDIGO_PLATEAU': 11,
}


class StageManager(object):
    def __init__(self):
        self.stage = 0
        self.stage_started = False
        self.blockings = []
        self.n_stage_started = 0
        self.n_stage_ended = 0

    def update(self, current_states):
        # current_states
        #   - items
        #   - map_id
        #   - badges
        #   - visited_pokecenters
        if self.stage >= len(STAGE_DICT):
            return
        stage = STAGE_DICT[self.stage]
        if not self.stage_started:
            if 'trigger' in stage and not self._check_trigger(stage['trigger'], current_states):
                return
            if not self.stage_started and 'start' in stage:
                self._start(stage['start'])
                self.stage_started = True
                self.n_stage_started += 1
                # print(f'stage started: {self.stage}')
        if self.stage_started:
            # if 'clear' in stage:
            assert 'clear' in stage, f'stage {self.stage} has no clear condition'
            if self._check_clear(stage['clear'], current_states):
                if 'end' in stage:
                    self._end(stage['end'])
                # print(f'stage cleared: {self.stage}')
                self.stage += 1
                self.stage_started = False
                self.n_stage_ended += 1
    
    def _check_trigger(self, trigger, current_states):
        if trigger[0] == 'have':
            return trigger[1] in current_states['items']
        elif trigger[0] == 'pokecenter':
            return POKECENTER_TO_INDEX_DICT[trigger[1]] in current_states['visited_pokecenters']
        elif trigger[0] == 'enter':
            return MAP_DICT[trigger[1]]['map_id'] == current_states['map_id']
        elif trigger[0] == 'last_pokecenter':
            return POKECENTER_TO_INDEX_DICT[trigger[1]] == current_states['last_pokecenter']
        else:
            raise NotImplementedError('trigger type not implemented: {}'.format(trigger[0]))
        
    def _start(self, start):
        for block in start:
            if block[0] == 'block':
                if block[1:] not in self.blockings:
                    self.blockings.append(block[1:])
            elif block[0] == 'unblock':
                self.blockings.remove(block[1:])
            else:
                raise NotImplementedError('start type not implemented: {}'.format(block[0]))
            
    def _check_clear(self, clear, current_states):
        # all clear conditions must be met
        for condition in clear:
            if condition[0] == 'badge':
                if current_states['badges'] < condition[1]:
                    return False
            elif condition[0] == 'have':
                if isinstance(condition[1], list):
                    if not any([item in current_states['items'] for item in condition[1]]):
                        return False
                else:
                    if condition[1] not in current_states['items']:
                        return False
            elif condition[0] == 'pokecenter':
                if POKECENTER_TO_INDEX_DICT[condition[1]] not in current_states['visited_pokecenters']:
                    return False
            elif condition[0] == 'event':
                if not current_states.get('events', {}).get(condition[1], False):
                    return False
            else:
                raise NotImplementedError('clear type not implemented: {}'.format(condition[0]))
        return True
    
    def _end(self, end):
        for block in end:
            if block[0] == 'unblock':
                self.blockings.remove(block[1:])
            elif block[0] == 'block':
                if block[1:] not in self.blockings:
                    self.blockings.append(block[1:])
            else:
                raise NotImplementedError('end type not implemented: {}'.format(block[0]))
