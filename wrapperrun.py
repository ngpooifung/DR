# from wrapperGrad import Gradwrapper
from wrapperRDR import RDRwrapper
# from wrapperVTDR import VTDRwrapper
# %%
# if torch.cuda.is_available():
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend = 'nccl')
#     device = torch.device('cuda', local_rank)
#     cudnn.deterministic = True
#     cudnn.benchmark = True
# else:
#     device = torch.device('cpu')
a,b,c,d = RDRwrapper('/home/pwuaj/data/RDRraw/test/1/STDR389-20170320@111304-L1-S.jpg')
print(a,b,c,d)
