class Logger:
    def __init__(self, max_schmidt_rank, max_num_points, max_num_unitaries, max_num_training_datasets):
        self.max_schmidt_rank = max_schmidt_rank
        self.max_num_points = max_num_points
        self.max_num_unitaries = max_num_unitaries
        self.max_num_training_datasets = max_num_training_datasets
        self.schmidt_rank = 0
        self.num_points = 0
        self.num_unitary = 0
        self.num_training_dataset = 0

    def update_schmidt_rank(self, schmidt_rank):
        self.schmidt_rank = schmidt_rank

    def update_num_points(self, num_points):
        self.num_points = num_points

    def update_num_unitary(self, num_unitary):
        self.num_unitary = num_unitary

    def update_num_training_dataset(self, num_training_dataset):
        self.num_training_dataset = num_training_dataset
        print(f"rank {self.schmidt_rank+1}/{self.max_schmidt_rank+1}: {2**self.schmidt_rank} ")
        print(f"num points {self.num_points+1}/{self.max_num_points}")
        print(f"unitary {self.num_unitary+1}/{self.max_num_unitaries}")
        print(f"training dataset {self.num_training_dataset+1}/{self.max_num_training_datasets}")
