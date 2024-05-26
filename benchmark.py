from avalanche.benchmarks.classic import CORe50

# creating the benchmark (scenario object)
perm_mnist = PermutedMNIST(
    n_experiences=3,
    seed=1234,
)

# recovering the train and test streams
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# iterating over the train stream
for experience in train_stream:
    print("Start of task ", experience.task_label)
    print('Classes in this task:', experience.classes_in_this_experience)

    # The current Pytorch training set can be easily recovered through the 
    # experience
    current_training_set = experience.dataset
    # ...as well as the task_label
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')

    # we can recover the corresponding test experience in the test stream
    current_test_set = test_stream[experience.current_experience].dataset
    print('This task contains', len(current_test_set), 'test examples')