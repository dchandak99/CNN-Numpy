from grade_layers import grade_layers
from grade_applications import grade_applications
from grade_trainer_train import grade_trainer_train
from check_gradients import check_fully_connected

marks = grade_layers() + check_fully_connected() + grade_trainer_train() + grade_applications()

print('Total Marks = ', marks)
