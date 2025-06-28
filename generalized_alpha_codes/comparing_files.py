import difflib

with open('/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/generalized_alpha_codes/main3D.py') as f1, open('/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/outputs/2025-06-11_18-15-18_complete/main3D_classv2.py') as f2:
    diff = difflib.unified_diff(
        f1.readlines(),
        f2.readlines(),
        fromfile='main3D.py',
        tofile='/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop/outputs/2025-06-11_18-15-18_complete/main3D_classv2.py',
    )
    print(''.join(diff))

