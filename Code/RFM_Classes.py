

class RFM_Inp:
    def __init__(self, 
                df=0.0,
                perc_decr_pt_ind=0.0,
                perc_decr_pt_nonind=0.0,
                perc_decr_pt_suog=0.0, 
                perc_decr_gr_onroad=0.0,
                rows=0.0,
                cols=0.0,
                polls = '',
                sens=0.0,
                run_name='',
                phi_j_onroad=0.0,
                phi_j_suog=0.0,
                phi_j_ind=0.0,
                phi_j_nonind=0.0,
                phi_k_onroad=0.0,
                phi_k_suog=0.0,
                phi_k_ind=0.0,
                phi_k_nonind=0.0
               ):
 
        self.df=df
        self.perc_decr_pt_ind=perc_decr_pt_ind
        self.perc_decr_pt_nonind=perc_decr_pt_nonind
        self.perc_decr_pt_suog=perc_decr_pt_suog
        self.perc_decr_gr_onroad=perc_decr_gr_onroad
        self.rows=rows
        self.cols=cols
        self.polls=polls
        self.sens=sens
        self.run_name=run_name
        self.phi_j_onroad=phi_j_onroad
        self.phi_j_suog=phi_j_suog
        self.phi_j_ind=phi_j_ind
        self.phi_j_nonind=phi_j_nonind
        self.phi_k_onroad=phi_k_onroad
        self.phi_k_suog=phi_k_suog
        self.phi_k_ind=phi_k_ind
        self.phi_k_nonind=phi_k_nonind

import numpy as np
class RFM_Inp_2:
    def __init__(self, 
                df=0.0,
                pert_fact_pt_ind=0.0,
                pert_fact_pt_nonind=0.0,
                pert_fact_pt_suog=0.0, 
                pert_fact_gr_onroad=0.0,
                rows=0.0,
                cols=0.0,
                polls=[],
                sens=0.0,
                run_name='',
                phi_j_onroad=0.0,
                phi_j_suog=0.0,
                phi_j_ind=0.0,
                phi_j_nonind=0.0,
                phi_k_onroad=0.0,
                phi_k_suog=0.0,
                phi_k_ind=0.0,
                phi_k_nonind=0.0,
                f1_pt_ind=0.0,
                f2_pt_ind=0.0,
                f1_pt_suog=0.0,
                f2_pt_suog=0.0,
                f1_pt_nonind=0.0,
                f2_pt_nonind=0.0,
                f1_gr_onroad=0.0,
                f2_gr_onroad=0.0
               ):
        self.df = df
        self.pert_fact_pt_ind = np.float64(pert_fact_pt_ind)
        self.pert_fact_pt_nonind = np.float64(pert_fact_pt_nonind)
        self.pert_fact_pt_suog = np.float64(pert_fact_pt_suog)
        self.pert_fact_gr_onroad = np.float64(pert_fact_gr_onroad)
        self.rows = np.float64(rows)
        self.cols = np.float64(cols)
        self.polls = polls
        self.sens = np.float64(sens)
        self.run_name = run_name
        self.phi_j_onroad = np.float64(phi_j_onroad)
        self.phi_j_suog = np.float64(phi_j_suog)
        self.phi_j_ind = np.float64(phi_j_ind)
        self.phi_j_nonind = np.float64(phi_j_nonind)
        self.phi_k_onroad = np.float64(phi_k_onroad)
        self.phi_k_suog = np.float64(phi_k_suog)
        self.phi_k_ind = np.float64(phi_k_ind)
        self.phi_k_nonind = np.float64(phi_k_nonind)
        self.f1_pt_ind = np.float64(f1_pt_ind)
        self.f2_pt_ind = np.float64(f2_pt_ind)
        self.f1_pt_suog = np.float64(f1_pt_suog)
        self.f2_pt_suog = np.float64(f2_pt_suog)
        self.f1_pt_nonind = np.float64(f1_pt_nonind)
        self.f2_pt_nonind = np.float64(f2_pt_nonind)
        self.f1_gr_onroad = np.float64(f1_gr_onroad)
        self.f2_gr_onroad = np.float64(f2_gr_onroad)

        
class Inp_Excel_Map:
    def __init__(self, 
                df='',
                col_name='',
                delta_row=0,
                delta_col=0, 
                breaks=[],
                colors=[],
                cell_ref_legend='',
                work_sheet='',
               ):   
        self.df=df
        self.col_name=col_name
        self.delta_row=delta_row
        self.delta_col=delta_col
        self.breaks=breaks
        self.colors=colors
        self.cell_ref_legend=cell_ref_legend
        self.work_sheet=work_sheet
       
