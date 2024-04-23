from wrapper import Gradwrapper, RDRwrapper, VTDRwrapper
a,b,c,d = Gradwrapper('/home/pwuaj/data/grad/training/1/STDR052-20180323@101729-L1-S.jpg')
print('model output:"%.2f"' %a,'threshold:"%.2f"' %b,c,'probability:"%.2f"' %d)
a,b,c,d = RDRwrapper('/home/pwuaj/data/RDRraw/test/1/STDR365-20180621@162653-L4-S.jpg')
print('model output:"%.2f"' %a,'threshold:"%.2f"' %b,c,'probability:"%.2f"' %d)
a,b,c,d = VTDRwrapper('/home/pwuaj/data/VTDRraw/test/0/STDR361-20180521@110114-R12-S.jpg')
print('model output:"%.2f"' %a,'threshold:"%.2f"' %b,c,'probability:"%.2f"' %d)
