x1 = [q_np1, u_nphalf, dP_g_np1_1st, dP_gamma_np1_1st, dP_N_np1_1st, dP_F_np1_1st];

phi1 = [qnp1-qn-dt/2*(qdot(t_n,q_n,u_nphalf)+qdot(t_np1,q_np1,u_nphalf))]