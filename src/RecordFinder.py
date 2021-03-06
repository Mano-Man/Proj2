# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import glob
import os

from Record import Mode, RecordType, load_from_file, gran_dict
import Config as cfg


# ----------------------------------------------------------------------------------------------------------------------
#                                     Util Functions for Finding Records
# ----------------------------------------------------------------------------------------------------------------------

class RecordFinder():
    def __init__(self, net_name, dataset_name, patch_size, ones_range, gran_thresh, max_acc_loss, init_acc,
                 pQ_ratio=cfg.PATCHQ_UPDATE_RATIO, cQ_ratio=cfg.CHANNELQ_UPDATE_RATIO,
                 lQ_option=cfg.LQ_OPTION, cQ_option=cfg.CQ_OPTION, pQ_option=cfg.PQ_OPTION):
        self.net_name = net_name
        self.dataset_name = dataset_name
        self.ps = patch_size
        self.ones_range = ones_range
        self.gran_thresh = gran_thresh
        self.max_acc_loss = max_acc_loss
        self.init_acc = init_acc
        self.pQ_ratio = pQ_ratio
        self.cQ_ratio = cQ_ratio
        self.lQ_option = lQ_option
        self.cQ_option = cQ_option
        self.pQ_option = pQ_option

    def find_rec_filename(self, mode, record_type):
        if record_type == RecordType.FIRST_LVL_REC:
            return self._find_rec_file_by_time(self._first_lvl_regex(mode))
        elif record_type == RecordType.cQ_REC:
            return self._find_rec_file_by_time(self._cQ_regex(mode))
        elif record_type == RecordType.lQ_RESUME:
            return self._find_rec_file_by_time(self._lQ_resume_regex(mode))
        elif record_type == RecordType.pQ_REC:
            return self._find_rec_file_by_time(self._pQ_regex(mode))
        elif record_type == RecordType.FINAL_RESULT_REC:
            return self._find_rec_file_by_time(self._final_rec_regex(mode))
        elif record_type == RecordType.BASELINE_REC:
            return self._find_rec_file_by_time(self._baseline_rec_regex())
        else:
            return None
        
    def find_all_recs_fns(self, mode, record_type):
        if record_type == RecordType.FIRST_LVL_REC:
            return glob.glob(os.path.join(cfg.RESULTS_DIR, self._first_lvl_regex(mode))) 
        elif record_type == RecordType.cQ_REC:
            return glob.glob(os.path.join(cfg.RESULTS_DIR, self._cQ_regex(mode))) 
        elif record_type == RecordType.lQ_RESUME:
            return glob.glob(os.path.join(cfg.RESULTS_DIR, self._lQ_resume_regex(mode))) 
        elif record_type == RecordType.pQ_REC:
            return glob.glob(os.path.join(cfg.RESULTS_DIR, self._pQ_regex(mode))) 
        elif record_type == RecordType.FINAL_RESULT_REC:
            return glob.glob(os.path.join(cfg.RESULTS_DIR, self._final_rec_regex(mode))) 
        elif record_type == RecordType.BASELINE_REC:
            return glob.glob(os.path.join(cfg.RESULTS_DIR, self._baseline_rec_regex())) 
        else:
            return None

    def find_all_FRs(self, mode):
        regex = self._final_rec_regex(mode)
        target = os.path.join(cfg.RESULTS_DIR, regex)
        return glob.glob(target)

    def print_result(self, mode):
        f_rec_fn = self.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
        if f_rec_fn is None:
            return
        f_rec = load_from_file(f_rec_fn, path='')
        print(f_rec)

    def _find_rec_file_by_time(self, regex):
        rec_filename = glob.glob(os.path.join(cfg.RESULTS_DIR, regex))
        if not rec_filename:
            return None
        else:
            rec_filename.sort(key=os.path.getmtime)
            return rec_filename[-1]

    def _first_lvl_regex(self, mode):
        gran_thresh = '*'
        if mode == Mode.MAX_GRANULARITY and type(self.gran_thresh) != str:
            gran_thresh = round(self.gran_thresh, 0)
        filename = (f'{self.net_name}_{self.dataset_name}_acc{self.init_acc}_{gran_dict[mode]}' +
                    f'_ps{self.ps}_ones{self.ones_range[0]}x{self.ones_range[1]}' +
                    f'_mg{gran_thresh}_*pkl')
        return filename

    def _cQ_regex(self, mode):
        regex = f'ChannelQ{self.cQ_option.value}r{self.cQ_ratio}_ma{self.max_acc_loss}_'
        if mode == Mode.UNIFORM_PATCH:
            regex += self._first_lvl_regex(mode)
        else:  # mode == rc.max_granularity
            regex += self._pQ_regex(mode)
        return regex

    def _lQ_resume_regex(self, mode):
        regex = f'LayerQ{self.lQ_option.value}_ma{self.max_acc_loss}_'
        if mode == Mode.UNIFORM_PATCH or mode == Mode.MAX_GRANULARITY:
            regex += self._cQ_regex(mode)
        elif mode == Mode.UNIFORM_FILTERS:
            regex += self._pQ_regex(mode)
        else:
            regex += self._first_lvl_regex(mode)
        return regex

    def _pQ_regex(self, mode):
        regex = f'PatchQ{self.pQ_option.value}r{self.pQ_ratio}_ma{self.max_acc_loss}_'
        return (regex + self._first_lvl_regex(mode))

    def _final_rec_regex(self, mode):
        quant_name = f'LQ{self.lQ_option.value}'
        if mode == Mode.UNIFORM_PATCH or mode == Mode.MAX_GRANULARITY:
            quant_name += f'_CQ{self.cQ_option.value}r{cfg.CHANNELQ_UPDATE_RATIO}'
        if mode == Mode.UNIFORM_FILTERS or mode == Mode.MAX_GRANULARITY:
            quant_name += f'_PQ{self.pQ_option.value}r{cfg.PATCHQ_UPDATE_RATIO}'
            
        reg = f'FR_{self.net_name}_{self.dataset_name}_acc{self.init_acc}_{quant_name}'
        reg+= f'_ps{self.ps}_ones{self.ones_range[0]}x{self.ones_range[1]}_'
        reg+= f'{gran_dict[mode]}_ma{self.max_acc_loss}_*pkl'
        return reg

    def _baseline_rec_regex(self):
        return f'BS_{self.net_name}_{self.dataset_name}_acc{self.init_acc}_ps{self.ps}_os*_bacc*pkl'

