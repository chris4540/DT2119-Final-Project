class Config:
    batch_size = 100
    n_epochs = 50
    init_lr = 1e-2  # this would not take effect as using cyclic lr
    momentum = 0.9
    weight_decay = 5e-4
    eta_min = 1e-5
    eta_max = 1e-2
    shuffle = True
    n_hidden_nodes = 78
    part_labeled = 0.3  # the percentage of labeled data
    n_features = 39
    n_classes = 48
    temp = 1
    teacher_tar_fmt = 'teacher_plbl{plbl}.tar'
    baseline_tar_fmt = 'baseline_plbl{plbl}.tar'
    student_tar_fmt = 'student_plbl{plbl}_T{temp}.tar'