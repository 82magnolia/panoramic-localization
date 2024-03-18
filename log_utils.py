import pickle


class PoseLogger:
    def __init__(self, log_dir, t_thres=0.1, r_thres=5):
        self.log_dir = log_dir
        self.pose_dict = {}  # Dictionary containing pose estimates and errors
        self.room_error_dict = {}  # Dictionary containing errors for each room
        self.split_error_dict = {}  # Dictionary containing errors for each split
        self.room_split_error_dict = {}  # Dictionary containing errors for each room/split
        self.stat_dict = {'total': {}}  # Dictionary containing pose estimate statistics
        self.skipped_rooms = []  # List containing skipped rooms
        self.total_error_dict = {'t_error_list': [], 'r_error_list': []}  # Dictionary containing total error
        
        # Thresholds for calculating accuracy
        self.t_thres = 0.1
        self.r_thres = 5
    
    def add_filename(self, filename, room_name, split='original'):
        self.pose_dict[(filename, room_name, split)] = {}
        if (room_name, split) not in self.stat_dict.keys():
            self.stat_dict[(room_name, split)] = {}
            self.room_split_error_dict[(room_name, split)] = {}
            self.room_split_error_dict[(room_name, split)]['t_error_list'] = []
            self.room_split_error_dict[(room_name, split)]['r_error_list'] = []
        
        if room_name not in self.stat_dict.keys():
            self.stat_dict[room_name] = {}
            self.room_error_dict[room_name] = {}
            self.room_error_dict[room_name]['t_error_list'] = []
            self.room_error_dict[room_name]['r_error_list'] = []

        if split not in self.stat_dict.keys():
            self.stat_dict[split] = {}
            self.split_error_dict[split] = {}
            self.split_error_dict[split]['t_error_list'] = []
            self.split_error_dict[split]['r_error_list'] = []

    def add_error(self, t_error, r_error, filename, room_name, split='original'):
        # t_error, r_error is assumed to be float
        self.pose_dict[(filename, room_name, split)]['t_error'] = t_error
        self.pose_dict[(filename, room_name, split)]['r_error'] = r_error
        self.room_split_error_dict[(room_name, split)]['t_error_list'].append(t_error)
        self.room_split_error_dict[(room_name, split)]['r_error_list'].append(r_error)
        self.room_error_dict[room_name]['t_error_list'].append(t_error)
        self.room_error_dict[room_name]['r_error_list'].append(r_error)
        self.split_error_dict[split]['t_error_list'].append(t_error)
        self.split_error_dict[split]['r_error_list'].append(r_error)
        self.total_error_dict['t_error_list'].append(t_error)
        self.total_error_dict['r_error_list'].append(r_error)

    def add_estimate(self, filename, translation, rotation, room_name, split='original'):
        # translation is assumed to be a (3, ) numpy array
        # rotation is assumed to be a (3, 3) numpy array 
        self.pose_dict[(filename, room_name, split)]['translation'] = translation
        self.pose_dict[(filename, room_name, split)]['rotation'] = rotation
    
    def add_skipped_room(self, filename):
        self.skipped_rooms.append(filename)
    
    def calc_statistics(self, target=None, filter_kwd_list=None):
        # target specifices which statistics will be computed
        all_names = self.pose_dict.keys()
        room_split_names = self.room_split_error_dict.keys()
        room_names = self.room_error_dict.keys()
        split_names = self.split_error_dict.keys()
        
        if target == 'room':
            for room in room_names:
                if len(self.room_error_dict[room]['t_error_list']) != 0:
                    self.stat_dict[room]['median_t_error'] = \
                        sorted(self.room_error_dict[room]['t_error_list'])[len(self.room_error_dict[room]['t_error_list']) // 2]
                    self.stat_dict[room]['median_r_error'] = \
                        sorted(self.room_error_dict[room]['r_error_list'])[len(self.room_error_dict[room]['r_error_list']) // 2]
                    self.stat_dict[room]['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                        in zip(self.room_error_dict[room]['t_error_list'], self.room_error_dict[room]['r_error_list'])])
                    self.stat_dict[room]['accuracy'] /= len(self.room_error_dict[room]['t_error_list'])
                else:
                    self.stat_dict[room]['median_t_error'] = 'N/A'
                    self.stat_dict[room]['median_r_error'] = 'N/A'
                    self.stat_dict[room]['accuracy'] = 'N/A'
        elif target == 'split':
            for split in split_names:
                if len(self.split_error_dict[split]['t_error_list']) != 0:
                    self.stat_dict[split]['median_t_error'] = \
                        sorted(self.split_error_dict[split]['t_error_list'])[len(self.split_error_dict[split]['t_error_list']) // 2]
                    self.stat_dict[split]['median_r_error'] = \
                        sorted(self.split_error_dict[split]['r_error_list'])[len(self.split_error_dict[split]['r_error_list']) // 2]
                    self.stat_dict[split]['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                        in zip(self.split_error_dict[split]['t_error_list'], self.split_error_dict[split]['r_error_list'])])
                    self.stat_dict[split]['accuracy'] /= len(self.split_error_dict[split]['t_error_list'])
                else:
                    self.stat_dict[split]['median_t_error'] = 'N/A'
                    self.stat_dict[split]['median_r_error'] = 'N/A'
                    self.stat_dict[split]['accuracy'] = 'N/A'
        elif target == 'room_split':
            for room_split in room_split_names:
                if len(self.room_split_error_dict[room_split]['t_error_list']) != 0:
                    self.stat_dict[room_split]['median_t_error'] = \
                        sorted(self.room_split_error_dict[room_split]['t_error_list'])[len(self.room_split_error_dict[room_split]['t_error_list']) // 2]
                    self.stat_dict[room_split]['median_r_error'] = \
                        sorted(self.room_split_error_dict[room_split]['r_error_list'])[len(self.room_split_error_dict[room_split]['r_error_list']) // 2]
                    self.stat_dict[room_split]['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                        in zip(self.room_split_error_dict[room_split]['t_error_list'], self.room_split_error_dict[room_split]['r_error_list'])])
                    self.stat_dict[room_split]['accuracy'] /= len(self.room_split_error_dict[room_split]['t_error_list'])
                else:
                    self.stat_dict[room_split]['median_t_error'] = 'N/A'
                    self.stat_dict[room_split]['median_r_error'] = 'N/A'
                    self.stat_dict[room_split]['accuracy'] = 'N/A'
        elif target == 'kwd':
            assert filter_kwd_list is not None
            self.kwd_dict = {'name': filter_kwd_list, 'median_t_error': None, 'median_r_error': None, 'accuracy': None, 't_error_list': [], 'r_error_list': []}
            for room_split in room_split_names:
                valid_room_split = len(self.room_split_error_dict[room_split]['t_error_list']) != 0
                fit_kwd = True
                for filter_kwd in filter_kwd_list:
                    if 'not' in filter_kwd:  # Exclusion query (e.g. not_area_1)
                        if not (filter_kwd.strip('not_') not in room_split[0] and filter_kwd.strip('not_') not in room_split[1]):
                            fit_kwd = False
                    else:  # Inclusion query (e.g. area_1)
                        if not (filter_kwd in room_split[0] or filter_kwd in room_split[1]):
                            fit_kwd = False

                if fit_kwd and valid_room_split:
                    self.kwd_dict['t_error_list'].extend(self.room_split_error_dict[room_split]['t_error_list'])
                    self.kwd_dict['r_error_list'].extend(self.room_split_error_dict[room_split]['r_error_list'])

            if len(self.kwd_dict['t_error_list']) != 0:
                self.kwd_dict['median_t_error'] = \
                    sorted(self.kwd_dict['t_error_list'])[len(self.kwd_dict['t_error_list']) // 2]
                self.kwd_dict['median_r_error'] = \
                    sorted(self.kwd_dict['r_error_list'])[len(self.kwd_dict['r_error_list']) // 2]
                self.kwd_dict['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                    in zip(self.kwd_dict['t_error_list'], self.kwd_dict['r_error_list'])])
                self.kwd_dict['accuracy'] /= len(self.kwd_dict['t_error_list'])
            else:
                self.kwd_dict['median_t_error'] = 'N/A'
                self.kwd_dict['median_r_error'] = 'N/A'
                self.kwd_dict['accuracy'] = 'N/A'
        elif target == 'total':
            self.stat_dict['total']['median_t_error'] = \
                sorted(self.total_error_dict['t_error_list'])[len(self.total_error_dict['t_error_list']) // 2]
            self.stat_dict['total']['median_r_error'] = \
                sorted(self.total_error_dict['r_error_list'])[len(self.total_error_dict['r_error_list']) // 2]
            self.stat_dict['total']['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                in zip(self.total_error_dict['t_error_list'], self.total_error_dict['r_error_list'])])
            self.stat_dict['total']['accuracy'] /= len(self.total_error_dict['t_error_list'])

    def print_stat(self, target):
        room_split_names = self.room_split_error_dict.keys()
        room_names = self.room_error_dict.keys()
        split_names = self.split_error_dict.keys()

        if target == 'total':
            print("Statistics for total:")
            print(self.stat_dict['total'])
        elif target == 'room':
            for room in room_names:
                print(f"Statistics for {room}:")
                print(self.stat_dict[room])
        elif target == 'split':
            for split in split_names:
                print(f"Statistics for {split}:")
                print(self.stat_dict[split])
        elif target == 'room_split':
            for room_split in room_split_names:
                print(f"Statistics for {room_split}:")
                print(self.stat_dict[room_split])
        elif target == 'kwd':
            print(f"Statistics for {self.kwd_dict['name']}")
            print(f"t-error: {self.kwd_dict['median_t_error']}, r-error: {self.kwd_dict['median_r_error']}, acc: {self.kwd_dict['accuracy']}")


def save_logger(pickle_name, logger: PoseLogger):
    pkl_file = open(pickle_name, 'wb')
    pickle.dump(logger, pkl_file)
    pkl_file.close()


def load_logger(pickle_name):
    pkl_file = open(pickle_name, 'rb')
    logger = pickle.load(pkl_file)
    pkl_file.close()
    return logger
