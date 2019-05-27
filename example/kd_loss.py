import torch
# from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax

if __name__=='__main__':
    logits1 = -np.random.rand(7,3)
    logits2 = -np.random.rand(5,3)

    logits3 = -np.random.rand(7,3)
    logits4 = -np.random.rand(5,3)

    #
    logits1 = torch.Tensor(logits1)
    logits2 = torch.Tensor(logits2)
    logits3 = torch.Tensor(logits3)
    logits4 = torch.Tensor(logits4)

    teacher_logits = pad_sequence([logits1, logits2])
    student_logits = pad_sequence([logits3, logits2])
    kd_loss = nn.KLDivLoss(reduce=False, reduction='none')(
            log_softmax(student_logits, dim=-1),
            softmax(teacher_logits, dim=-1))
    print(kd_loss)

    # kd_loss = nn.KLDivLoss(reduction='batchmean')(
    #         log_softmax(student_logits, dim=-1),
    #         softmax(teacher_logits, dim=-1))
    # print(kd_loss)

    # kd_loss = nn.KLDivLoss(reduction='sum')(
    #         log_softmax(student_logits, dim=-1),
    #         softmax(teacher_logits, dim=-1))
    # print(kd_loss)