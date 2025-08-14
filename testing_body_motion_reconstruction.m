
OR_filename = '/Users/theresahonein/Library/CloudStorage/GoogleDrive-terryhonein@gmail.com/.shortcut-targets-by-id/1YxSyglmRD1uqwzqgPG_BfjlTJtQipxSq/Hula Hoop/2025-08-12 Experiment 3/2025-08-12 Euler Angles/OR_20250812_113440.csv';
OL_filename = '/Users/theresahonein/Library/CloudStorage/GoogleDrive-terryhonein@gmail.com/.shortcut-targets-by-id/1YxSyglmRD1uqwzqgPG_BfjlTJtQipxSq/Hula Hoop/2025-08-12 Experiment 3/2025-08-12 Euler Angles/OL_20250812_113440.csv';
IB_filename = '/Users/theresahonein/Library/CloudStorage/GoogleDrive-terryhonein@gmail.com/.shortcut-targets-by-id/1YxSyglmRD1uqwzqgPG_BfjlTJtQipxSq/Hula Hoop/2025-08-12 Experiment 3/2025-08-12 Euler Angles/IB_20250812_113440.csv';
IT_filename = '/Users/theresahonein/Library/CloudStorage/GoogleDrive-terryhonein@gmail.com/.shortcut-targets-by-id/1YxSyglmRD1uqwzqgPG_BfjlTJtQipxSq/Hula Hoop/2025-08-12 Experiment 3/2025-08-12 Euler Angles/IT_20250812_113440.csv';
IL_filename = '/Users/theresahonein/Library/CloudStorage/GoogleDrive-terryhonein@gmail.com/.shortcut-targets-by-id/1YxSyglmRD1uqwzqgPG_BfjlTJtQipxSq/Hula Hoop/2025-08-12 Experiment 3/2025-08-12 Euler Angles/IL_20250812_113440.csv';
    
[OR_psi,OR_theta,OR_phi,OR_dx,OR_dy,OR_dz] = read_data(OR_filename);
[OL_psi,OL_theta,OL_phi,OL_dx,OL_dy,OL_dz] = read_data(OL_filename);
[IB_psi,IB_theta,IB_phi,IB_dx,IB_dy,IB_dz] = read_data(IB_filename);

function [psi,theta,phi,dx,dy,dz] = read_data(filename)
    % Specify your file

    data = readtable(filename, 'Delimiter', ',');
    
    psi = data.Euler_X;
    theta = data.Euler_Y;
    phi = data.Euler_Z;
    
    dx = data.dq_X;
    dy = data.dq_Y;
    dz = data.dq_Z;
end