# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:12:03 2014

@author: Jm Begon
"""
import numpy as np
from util import plotFeatureImportance


def plotFI(data, tNum):
    plotFeatureImportance(data, "RandConv with original image (feature 0) \nTest number "+str(tNum), None)

if __name__ == "__main__":
    usual = [ 0.0271098, 0.0189606, 0.0001375, 0.0236002, 0.0107521, 0.0015851, 0.0081692, 0.0190917, 0.0116396, 0.0004176, 0.0011001, 0.0016647, 0.0194823, 0.0087727, 0.0004605, 0.0131183, 0.0000099, 0.0171307, 0.0104943, 0.0217925, 0.0061458, 0.0000000, 0.0103114, 0.0177175, 0.0099031, 0.0028929, 0.0154457, 0.0271596, 0.0000196, 0.0104398, 0.0222576, 0.0010369, 0.0170559, 0.0000103, 0.0127365, 0.0085806, 0.0273544, 0.0175444, 0.0000023, 0.0000069, 0.0001545, 0.0000285, 0.0000121, 0.0205825, 0.0146406, 0.0282398, 0.0054543, 0.0161403, 0.0038054, 0.0165724, 0.0247801, 0.0113244, 0.0164987, 0.0025082, 0.0000017, 0.0076390, 0.0272782, 0.0175415, 0.0113167, 0.0165124, 0.0100562, 0.0218732, 0.0000072, 0.0093629, 0.0014318, 0.0180456, 0.0173523, 0.0000178, 0.0020570, 0.0046616, 0.0025479, 0.0063238, 0.0046936, 0.0011504, 0.0063963, 0.0190344, 0.0018903, 0.0130071, 0.0000473, 0.0219228, 0.0196532, 0.0258176, 0.0218472, 0.0000175, 0.0001266, 0.0026593, 0.0045112, 0.0187227, 0.0156148, 0.0000059, 0.0095575, 0.0214169, 0.0060630, 0.0051031, 0.0010056, 0.0093537, 0.0011318, 0.0013847, 0.0053247, 0.0014987, 0.0041633, ]
    t1 = [ 0.0279026, 0.0189900, 0.0001346, 0.0237569, 0.0108788, 0.0015828, 0.0083004, 0.0189099, 0.0116081, 0.0004357, 0.0010643, 0.0016963, 0.0194178, 0.0088962, 0.0004616, 0.0129674, 0.0000061, 0.0166708, 0.0103970, 0.0208908, 0.0061848, 0.0000000, 0.0104735, 0.0181949, 0.0099536, 0.0029615, 0.0152234, 0.0261952, 0.0000219, 0.0106145, 0.0214273, 0.0010650, 0.0168798, 0.0000083, 0.0127245, 0.0085923, 0.0276967, 0.0177476, 0.0000024, 0.0000070, 0.0001381, 0.0000348, 0.0000125, 0.0204437, 0.0146638, 0.0294540, 0.0053933, 0.0158763, 0.0038510, 0.0166609, 0.0249083, 0.0116044, 0.0165735, 0.0024524, 0.0000019, 0.0079817, 0.0235185, 0.0175811, 0.0115271, 0.0165300, 0.0101537, 0.0211358, 0.0000067, 0.0094493, 0.0014000, 0.0181236, 0.0172645, 0.0000157, 0.0020671, 0.0046683, 0.0025470, 0.0063906, 0.0047384, 0.0011267, 0.0064385, 0.0193040, 0.0019671, 0.0132735, 0.0000480, 0.0218550, 0.0195263, 0.0291050, 0.0229945, 0.0000146, 0.0001321, 0.0027090, 0.0045056, 0.0186799, 0.0151915, 0.0000054, 0.0095787, 0.0204212, 0.0060202, 0.0050420, 0.0009910, 0.0093637, 0.0011327, 0.0013884, 0.0053872, 0.0014878, 0.0042003, ]
    t2 = [ 0.0297412, 0.0189376, 0.0001364, 0.0242624, 0.0107417, 0.0016227, 0.0082982, 0.0189432, 0.0121014, 0.0004070, 0.0010903, 0.0017121, 0.0194547, 0.0087041, 0.0004456, 0.0130047, 0.0000078, 0.0165120, 0.0104871, 0.0237135, 0.0061505, 0.0000001, 0.0103287, 0.0179579, 0.0098705, 0.0029767, 0.0153233, 0.0250681, 0.0000213, 0.0104010, 0.0215517, 0.0010323, 0.0172694, 0.0000135, 0.0127804, 0.0085917, 0.0274492, 0.0176529, 0.0000012, 0.0000059, 0.0001419, 0.0000336, 0.0000152, 0.0212622, 0.0144163, 0.0273110, 0.0054418, 0.0159692, 0.0037405, 0.0165387, 0.0249421, 0.0114393, 0.0164268, 0.0025039, 0.0000037, 0.0077636, 0.0229300, 0.0177649, 0.0115860, 0.0165651, 0.0100234, 0.0226135, 0.0000081, 0.0093999, 0.0014394, 0.0180808, 0.0174791, 0.0000163, 0.0020743, 0.0046764, 0.0024568, 0.0062996, 0.0048287, 0.0011628, 0.0063353, 0.0191521, 0.0019409, 0.0129858, 0.0000453, 0.0218721, 0.0197677, 0.0263662, 0.0222203, 0.0000160, 0.0001292, 0.0027019, 0.0045512, 0.0183289, 0.0160172, 0.0000068, 0.0095474, 0.0208431, 0.0060225, 0.0050943, 0.0010227, 0.0093866, 0.0011106, 0.0014207, 0.0053518, 0.0015190, 0.0041193, ]
    t3 = [ 0.0298486, 0.0192353, 0.0001229, 0.0229782, 0.0107492, 0.0016635, 0.0083339, 0.0193689, 0.0113916, 0.0004310, 0.0011097, 0.0016867, 0.0192452, 0.0087426, 0.0004581, 0.0130755, 0.0000080, 0.0165609, 0.0103700, 0.0220249, 0.0061484, 0.0000000, 0.0105537, 0.0180249, 0.0099070, 0.0029142, 0.0159714, 0.0274895, 0.0000208, 0.0106438, 0.0214911, 0.0010421, 0.0172270, 0.0000128, 0.0128503, 0.0086481, 0.0267420, 0.0175557, 0.0000020, 0.0000059, 0.0001449, 0.0000313, 0.0000153, 0.0206571, 0.0146054, 0.0276959, 0.0054629, 0.0167208, 0.0037755, 0.0163933, 0.0246697, 0.0113834, 0.0165201, 0.0024799, 0.0000034, 0.0078866, 0.0250571, 0.0174993, 0.0114556, 0.0164518, 0.0100722, 0.0227377, 0.0000068, 0.0090646, 0.0013857, 0.0178441, 0.0171830, 0.0000193, 0.0020540, 0.0046258, 0.0025000, 0.0063300, 0.0046728, 0.0011429, 0.0063808, 0.0194647, 0.0019444, 0.0129303, 0.0000449, 0.0215060, 0.0199818, 0.0254335, 0.0232162, 0.0000134, 0.0001171, 0.0027077, 0.0045734, 0.0187899, 0.0152505, 0.0000054, 0.0096322, 0.0198548, 0.0060844, 0.0050518, 0.0009807, 0.0093151, 0.0011503, 0.0013978, 0.0053495, 0.0015077, 0.0041123, ]
    t4 = [ 0.0270157, 0.0187720, 0.0001332, 0.0246264, 0.0108748, 0.0016045, 0.0082566, 0.0191139, 0.0114654, 0.0004222, 0.0011049, 0.0017120, 0.0195364, 0.0086935, 0.0004623, 0.0131553, 0.0000057, 0.0168470, 0.0103746, 0.0231926, 0.0061790, 0.0000000, 0.0103290, 0.0180588, 0.0099965, 0.0029630, 0.0154520, 0.0274747, 0.0000221, 0.0105601, 0.0212759, 0.0010319, 0.0170807, 0.0000117, 0.0128251, 0.0085891, 0.0279094, 0.0177093, 0.0000021, 0.0000061, 0.0001351, 0.0000319, 0.0000144, 0.0205052, 0.0144947, 0.0271180, 0.0053799, 0.0164092, 0.0038418, 0.0166442, 0.0242243, 0.0114834, 0.0164384, 0.0024970, 0.0000019, 0.0079474, 0.0233937, 0.0177434, 0.0113931, 0.0165890, 0.0101373, 0.0218014, 0.0000058, 0.0092277, 0.0014427, 0.0177859, 0.0171240, 0.0000180, 0.0020664, 0.0045806, 0.0025349, 0.0063984, 0.0047424, 0.0011609, 0.0063694, 0.0193960, 0.0019044, 0.0130194, 0.0000445, 0.0215432, 0.0200102, 0.0283256, 0.0240717, 0.0000132, 0.0001224, 0.0026982, 0.0046321, 0.0175395, 0.0158224, 0.0000070, 0.0096759, 0.0195938, 0.0060903, 0.0050850, 0.0009839, 0.0092929, 0.0011451, 0.0014013, 0.0054032, 0.0014893, 0.0041625, ]
    t5 = [ 0.0275202, 0.0187076, 0.0001294, 0.0236951, 0.0107777, 0.0016162, 0.0082922, 0.0189812, 0.0115656, 0.0004117, 0.0010857, 0.0016477, 0.0194503, 0.0088522, 0.0004527, 0.0130670, 0.0000052, 0.0161532, 0.0104447, 0.0223470, 0.0061210, 0.0000000, 0.0104872, 0.0178652, 0.0099113, 0.0029468, 0.0152930, 0.0251780, 0.0000207, 0.0105746, 0.0217059, 0.0010619, 0.0172121, 0.0000108, 0.0127859, 0.0086156, 0.0257677, 0.0176695, 0.0000012, 0.0000060, 0.0001455, 0.0000332, 0.0000163, 0.0204326, 0.0144830, 0.0288992, 0.0054079, 0.0163788, 0.0037629, 0.0176842, 0.0270334, 0.0112183, 0.0163860, 0.0024534, 0.0000026, 0.0076331, 0.0233320, 0.0174817, 0.0112543, 0.0165653, 0.0102655, 0.0215789, 0.0000072, 0.0092898, 0.0014303, 0.0183744, 0.0177090, 0.0000242, 0.0020326, 0.0047126, 0.0025593, 0.0062835, 0.0047460, 0.0011607, 0.0063564, 0.0194809, 0.0019501, 0.0130373, 0.0000507, 0.0217570, 0.0199947, 0.0273937, 0.0229228, 0.0000134, 0.0001304, 0.0026735, 0.0045778, 0.0179870, 0.0159606, 0.0000053, 0.0096628, 0.0216632, 0.0060623, 0.0051228, 0.0010143, 0.0093416, 0.0011274, 0.0013834, 0.0054279, 0.0015015, 0.0041863, ]
    t6 = [ 0.0274048, 0.0190568, 0.0001342, 0.0224939, 0.0106572, 0.0016027, 0.0081995, 0.0192050, 0.0117573, 0.0004177, 0.0011068, 0.0016903, 0.0193543, 0.0086441, 0.0004590, 0.0131242, 0.0000076, 0.0166364, 0.0105257, 0.0232472, 0.0061284, 0.0000000, 0.0104319, 0.0179491, 0.0098835, 0.0029123, 0.0151523, 0.0258061, 0.0000222, 0.0106690, 0.0219460, 0.0010351, 0.0176618, 0.0000087, 0.0127320, 0.0086418, 0.0263763, 0.0176653, 0.0000008, 0.0000056, 0.0001378, 0.0000312, 0.0000179, 0.0209119, 0.0145756, 0.0292111, 0.0054452, 0.0161906, 0.0038777, 0.0168128, 0.0263249, 0.0112360, 0.0164176, 0.0024556, 0.0000020, 0.0073508, 0.0238467, 0.0178117, 0.0115023, 0.0164228, 0.0102249, 0.0220412, 0.0000069, 0.0092506, 0.0014533, 0.0180110, 0.0173003, 0.0000160, 0.0020715, 0.0046487, 0.0024287, 0.0063432, 0.0047731, 0.0011371, 0.0064309, 0.0190841, 0.0019291, 0.0134287, 0.0000444, 0.0216693, 0.0201375, 0.0265172, 0.0232175, 0.0000158, 0.0001332, 0.0027232, 0.0045763, 0.0183636, 0.0153289, 0.0000086, 0.0097404, 0.0207101, 0.0060635, 0.0049987, 0.0009847, 0.0092791, 0.0011337, 0.0013892, 0.0053862, 0.0015069, 0.0041574, ]
    t7 = [ 0.0291776, 0.0193564, 0.0001257, 0.0227833, 0.0107687, 0.0015826, 0.0081156, 0.0191187, 0.0118003, 0.0004157, 0.0010848, 0.0016867, 0.0196791, 0.0087986, 0.0004649, 0.0130731, 0.0000075, 0.0170518, 0.0103025, 0.0235111, 0.0061079, 0.0000000, 0.0103901, 0.0178920, 0.0099017, 0.0029443, 0.0151863, 0.0258771, 0.0000166, 0.0104653, 0.0218682, 0.0010433, 0.0168433, 0.0000106, 0.0127178, 0.0086039, 0.0276602, 0.0177167, 0.0000028, 0.0000059, 0.0001404, 0.0000305, 0.0000138, 0.0212797, 0.0145528, 0.0294467, 0.0053809, 0.0157589, 0.0037967, 0.0166557, 0.0247163, 0.0112752, 0.0165825, 0.0024804, 0.0000032, 0.0078232, 0.0238542, 0.0175643, 0.0117048, 0.0165184, 0.0101700, 0.0213427, 0.0000067, 0.0091739, 0.0014150, 0.0182159, 0.0169912, 0.0000180, 0.0020774, 0.0045629, 0.0025121, 0.0063530, 0.0047493, 0.0011482, 0.0063707, 0.0196491, 0.0019348, 0.0132102, 0.0000444, 0.0215754, 0.0197279, 0.0266328, 0.0217061, 0.0000168, 0.0001298, 0.0026867, 0.0045307, 0.0176708, 0.0156829, 0.0000062, 0.0096671, 0.0212406, 0.0060315, 0.0051192, 0.0010161, 0.0093216, 0.0011282, 0.0013796, 0.0054214, 0.0014968, 0.0041287, ]
    t8 = [ 0.0295453, 0.0192092, 0.0001194, 0.0221435, 0.0108398, 0.0015821, 0.0083786, 0.0193443, 0.0114721, 0.0004164, 0.0011057, 0.0016581, 0.0194587, 0.0086278, 0.0004630, 0.0130416, 0.0000061, 0.0166635, 0.0104216, 0.0215338, 0.0061006, 0.0000000, 0.0103589, 0.0179178, 0.0099680, 0.0029158, 0.0158415, 0.0271169, 0.0000215, 0.0104126, 0.0212024, 0.0010480, 0.0177604, 0.0000085, 0.0125444, 0.0087052, 0.0290046, 0.0180690, 0.0000021, 0.0000072, 0.0001395, 0.0000306, 0.0000136, 0.0209126, 0.0145973, 0.0282748, 0.0054717, 0.0161830, 0.0038208, 0.0172807, 0.0262819, 0.0112012, 0.0163847, 0.0024550, 0.0000016, 0.0078312, 0.0226959, 0.0175860, 0.0116458, 0.0164668, 0.0101240, 0.0212059, 0.0000082, 0.0091720, 0.0014360, 0.0184305, 0.0170972, 0.0000200, 0.0021132, 0.0046124, 0.0024927, 0.0062482, 0.0047321, 0.0011403, 0.0064185, 0.0190555, 0.0019318, 0.0130759, 0.0000452, 0.0216742, 0.0196335, 0.0260675, 0.0222016, 0.0000180, 0.0001297, 0.0027313, 0.0046657, 0.0183920, 0.0154205, 0.0000068, 0.0096203, 0.0206335, 0.0060647, 0.0051473, 0.0009794, 0.0093830, 0.0011300, 0.0013812, 0.0053912, 0.0014767, 0.0041818, ]
    t9 = [ 0.0286185, 0.0191308, 0.0001356, 0.0238166, 0.0109287, 0.0015779, 0.0083724, 0.0192509, 0.0116706, 0.0004382, 0.0010837, 0.0016730, 0.0194778, 0.0087166, 0.0004516, 0.0130808, 0.0000065, 0.0166013, 0.0104805, 0.0227471, 0.0061215, 0.0000001, 0.0102778, 0.0178734, 0.0098751, 0.0029244, 0.0155229, 0.0257196, 0.0000202, 0.0103650, 0.0219640, 0.0010485, 0.0167919, 0.0000122, 0.0126033, 0.0086592, 0.0273587, 0.0176526, 0.0000025, 0.0000057, 0.0001466, 0.0000325, 0.0000128, 0.0205757, 0.0144006, 0.0275649, 0.0054295, 0.0162444, 0.0037701, 0.0172131, 0.0262038, 0.0113213, 0.0164976, 0.0024905, 0.0000023, 0.0076222, 0.0234621, 0.0175816, 0.0112916, 0.0165721, 0.0101501, 0.0231623, 0.0000084, 0.0094782, 0.0014026, 0.0177368, 0.0172442, 0.0000189, 0.0020006, 0.0046564, 0.0025000, 0.0064172, 0.0048020, 0.0011690, 0.0063498, 0.0194620, 0.0019329, 0.0131046, 0.0000435, 0.0221578, 0.0200672, 0.0264990, 0.0218023, 0.0000155, 0.0001202, 0.0027145, 0.0045230, 0.0176244, 0.0156864, 0.0000058, 0.0096319, 0.0210441, 0.0059954, 0.0050688, 0.0009937, 0.0092697, 0.0011365, 0.0014038, 0.0053668, 0.0015223, 0.0042184, ]
    t10 = [ 0.0279407, 0.0187087, 0.0001411, 0.0236984, 0.0108953, 0.0016054, 0.0080860, 0.0191811, 0.0119471, 0.0004177, 0.0010753, 0.0017106, 0.0194980, 0.0088737, 0.0004484, 0.0131189, 0.0000053, 0.0166589, 0.0104468, 0.0225765, 0.0062208, 0.0000001, 0.0104395, 0.0178097, 0.0098806, 0.0029586, 0.0148634, 0.0276013, 0.0000214, 0.0105288, 0.0216066, 0.0010476, 0.0180206, 0.0000115, 0.0128641, 0.0085551, 0.0265532, 0.0176001, 0.0000024, 0.0000080, 0.0001330, 0.0000307, 0.0000140, 0.0203715, 0.0146193, 0.0292104, 0.0053521, 0.0161456, 0.0038094, 0.0168242, 0.0265762, 0.0114568, 0.0163973, 0.0024357, 0.0000021, 0.0076924, 0.0234581, 0.0176762, 0.0113115, 0.0164813, 0.0101119, 0.0228208, 0.0000070, 0.0092565, 0.0014348, 0.0181636, 0.0171604, 0.0000214, 0.0020548, 0.0046749, 0.0024760, 0.0062107, 0.0047900, 0.0011570, 0.0064538, 0.0191093, 0.0019199, 0.0128892, 0.0000414, 0.0221481, 0.0201879, 0.0252969, 0.0220609, 0.0000197, 0.0001275, 0.0027079, 0.0045103, 0.0182022, 0.0152227, 0.0000055, 0.0096516, 0.0202602, 0.0060789, 0.0051366, 0.0010142, 0.0094545, 0.0011246, 0.0013945, 0.0053328, 0.0014955, 0.0041885, ]
    t11 = [ 0.0297412, 0.0189376, 0.0001364, 0.0242624, 0.0107417, 0.0016227, 0.0082982, 0.0189432, 0.0121014, 0.0004070, 0.0010903, 0.0017121, 0.0194547, 0.0087041, 0.0004456, 0.0130047, 0.0000078, 0.0165120, 0.0104871, 0.0237135, 0.0061505, 0.0000001, 0.0103287, 0.0179579, 0.0098705, 0.0029767, 0.0153233, 0.0250681, 0.0000213, 0.0104010, 0.0215517, 0.0010323, 0.0172694, 0.0000135, 0.0127804, 0.0085917, 0.0274492, 0.0176529, 0.0000012, 0.0000059, 0.0001419, 0.0000336, 0.0000152, 0.0212622, 0.0144163, 0.0273110, 0.0054418, 0.0159692, 0.0037405, 0.0165387, 0.0249421, 0.0114393, 0.0164268, 0.0025039, 0.0000037, 0.0077636, 0.0229300, 0.0177649, 0.0115860, 0.0165651, 0.0100234, 0.0226135, 0.0000081, 0.0093999, 0.0014394, 0.0180808, 0.0174791, 0.0000163, 0.0020743, 0.0046764, 0.0024568, 0.0062996, 0.0048287, 0.0011628, 0.0063353, 0.0191521, 0.0019409, 0.0129858, 0.0000453, 0.0218721, 0.0197677, 0.0263662, 0.0222203, 0.0000160, 0.0001292, 0.0027019, 0.0045512, 0.0183289, 0.0160172, 0.0000068, 0.0095474, 0.0208431, 0.0060225, 0.0050943, 0.0010227, 0.0093866, 0.0011106, 0.0014207, 0.0053518, 0.0015190, 0.0041193, ]

    matrix = np.vstack([usual, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11])
    corrMat = np.corrcoef(matrix)
    print corrMat