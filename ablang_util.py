

import os
import sys
import argparse
import copy

import numpy as np
import torch
import ablang

class ABlangEmbedding():
    def __init__(self, hltype, device):
        if hltype in ["heavy", "light"]:
            self.model = ablang.pretrained(hltype)
        else:
            return "## chech hltype parameter"
        self.device = device
        self.model.AbLang = self.model.AbLang.eval()
        self.model.AbLang = self.model.AbLang.to(self.device)
        self.tokenizer = self.model.tokenizer
        self.mask_ids = self.tokenizer.vocab_to_token["*"]
    
    def soembed( self, sequence):
        sequence = sequence.replace("-","")
        self.model.AbLang.AbRep.eval()
        ids = self.tokenizer( [sequence], pad=True ).to(self.device)
        with torch.no_grad():
            so = self.model.AbLang.AbRep(ids).last_hidden_states[0]
        so = so.to("cpu")
        so_mean = so[1:len(so)-1].mean(0).to("cpu")
        po = so[0]
        return so_mean, so, po

    def MLMporb(self, ids):
        ids = ids.to(self.device)
        with torch.no_grad():
            mlhead = self.model.AbLang( ids )[0]
            prob = torch.softmax( mlhead, dim=1)
            prob = prob.to("cpu").numpy()
        return prob
    
    def MLMscore( self, sequence ):
        sequence = sequence.replace("-", "")
        ids = self.model.tokenizer([sequence], pad=True).to(self.device)
        probs = []
        probs_all = []
        for pos in range( len(sequence)):
            aa = sequence[pos]
            aa_ids = self.tokenizer.vocab_to_token[aa]
            ids_ = copy.deepcopy(ids)
            prob = self.MLMporb( ids_ )
            prob_aa = prob[ pos+1 ][ aa_ids ]
            probs.append( prob_aa )
            probs_all.append( prob[pos+1] )
        probs = np.array( probs )
        probs_all = np.array( probs_all )
        nlls = -np.log10( probs )
        mlmscore = np.sum( nlls )
        return mlmscore, probs, probs_all

if __name__ == "__main__":
    device = "cpu"
    modelh = ABlangEmbedding( "heavy", device)
    # modell = ABlangEmbedding( "light", device)

    sequence = "AAAAAAAA"
    # sequence = "VQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARXXYYYDSSGXXXXYFDYWGQGTLVTVSS"
    soh, _, _ = modelh.soembed( sequence )
    sol, _, _ = modell.soembed( sequence )

    scoreh, probh, _ = modelh.MLMscore( sequence )
    scorel, probl, _ = modell.MLMscore( sequence )

    # import sys
    # sys.path.append("./")