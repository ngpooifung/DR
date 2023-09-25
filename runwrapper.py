from wrapper import Gradwrapper, RDRwrapper, VTDRwrapper
a,b,c,d = Gradwrapper('/home/pwuaj/data/grad/training/1/STDR052-20180323@101729-L1-S.jpg')
print(a,b,c,d)
a,b,c,d = Gradwrapper('/home/pwuaj/data/grad/validation/1/STDR052-20170911@105117-L1-S.jpg')
print(a,b,c,d)
