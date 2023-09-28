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
    
    def calc_statistics(self, target=None):
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
                self.stat_dict[split]['median_t_error'] = \
                    sorted(self.split_error_dict[split]['t_error_list'])[len(self.split_error_dict[split]['t_error_list']) // 2]
                self.stat_dict[split]['median_r_error'] = \
                    sorted(self.split_error_dict[split]['r_error_list'])[len(self.split_error_dict[split]['r_error_list']) // 2]
                self.stat_dict[split]['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                    in zip(self.split_error_dict[split]['t_error_list'], self.split_error_dict[split]['r_error_list'])])
                self.stat_dict[split]['accuracy'] /= len(self.split_error_dict[split]['t_error_list'])
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


class PerspectivePoseLogger:
    def __init__(self, log_dir, t_thres=0.1, r_thres=5):
        self.log_dir = log_dir
        self.pose_dict = {}  # Dictionary containing pose estimates and errors
        self.scene_error_dict = {}  # Dictionary containing errors for each scene
        self.stat_dict = {'total': {}}  # Dictionary containing pose estimate statistics
        self.skipped_imgs = []  # List containing skipped images
        self.total_error_dict = {'t_error_list': [], 'r_error_list': []}  # Dictionary containing total error
        
        # Thresholds for calculating accuracy
        self.t_thres = 0.1
        self.r_thres = 5
    
    def add_filename(self, filename, scene_name):
        self.pose_dict[filename] = {}

        if scene_name not in self.stat_dict.keys():
            self.stat_dict[scene_name] = {}
            self.scene_error_dict[scene_name] = {}
            self.scene_error_dict[scene_name]['t_error_list'] = []
            self.scene_error_dict[scene_name]['r_error_list'] = []

    def add_error(self, t_error, r_error, filename, scene_name):
        # t_error, r_error is assumed to be float
        self.pose_dict[filename]['t_error'] = t_error
        self.pose_dict[filename]['r_error'] = r_error
        self.scene_error_dict[scene_name]['t_error_list'].append(t_error)
        self.scene_error_dict[scene_name]['r_error_list'].append(r_error)
        self.total_error_dict['t_error_list'].append(t_error)
        self.total_error_dict['r_error_list'].append(r_error)

    def add_estimate(self, filename, translation, rotation):
        # translation is assumed to be a (3, ) numpy array
        # rotation is assumed to be a (3, 3) numpy array 
        self.pose_dict[filename]['translation'] = translation
        self.pose_dict[filename]['rotation'] = rotation
    
    def add_skipped_room(self, filename):
        self.skipped_imgs.append(filename)
    
    def calc_statistics(self, target=None):
        # target specifices which statistics will be computed
        scene_names = self.scene_error_dict.keys()
        
        if target == 'scene':
            print("Calculating Room Statistics")
            for scene in scene_names:
                if len(self.scene_error_dict[scene]['t_error_list']) != 0:
                    self.stat_dict[scene]['median_t_error'] = \
                        sorted(self.scene_error_dict[scene]['t_error_list'])[len(self.scene_error_dict[scene]['t_error_list']) // 2]
                    self.stat_dict[scene]['median_r_error'] = \
                        sorted(self.scene_error_dict[scene]['r_error_list'])[len(self.scene_error_dict[scene]['r_error_list']) // 2]
                    self.stat_dict[scene]['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                        in zip(self.scene_error_dict[scene]['t_error_list'], self.scene_error_dict[scene]['r_error_list'])])
                    self.stat_dict[scene]['accuracy'] /= len(self.scene_error_dict[scene]['t_error_list'])
                else:
                    self.stat_dict[scene]['median_t_error'] = 'N/A'
                    self.stat_dict[scene]['median_r_error'] = 'N/A'
                    self.stat_dict[scene]['accuracy'] = 'N/A'
        elif target == 'total':
            print("Calculating Total Statistics")
            self.stat_dict['total']['median_t_error'] = \
                sorted(self.total_error_dict['t_error_list'])[len(self.total_error_dict['t_error_list']) // 2]
            self.stat_dict['total']['median_r_error'] = \
                sorted(self.total_error_dict['r_error_list'])[len(self.total_error_dict['r_error_list']) // 2]
            self.stat_dict['total']['accuracy'] = sum([(t_error < self.t_thres and r_error < self.r_thres) for (t_error, r_error)
                in zip(self.total_error_dict['t_error_list'], self.total_error_dict['r_error_list'])])
            self.stat_dict['total']['accuracy'] /= len(self.total_error_dict['t_error_list'])

    def print_stat(self, target):
        scene_names = self.scene_error_dict.keys()

        if target == 'total':
            print("Statistics for total:")
            print(self.stat_dict['total'])
        elif target == 'scene':
            for scene in scene_names:
                print(f"Statistics for {scene}:")
                print(self.stat_dict[scene])


def save_logger(pickle_name, logger: PoseLogger):
    pkl_file = open(pickle_name, 'wb')
    pickle.dump(logger, pkl_file)
    pkl_file.close()


def load_logger(pickle_name):
    pkl_file = open(pickle_name, 'rb')
    logger = pickle.load(pkl_file)
    pkl_file.close()
    return logger
