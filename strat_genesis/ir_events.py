from foolbox.api import *
from matplotlib.backends.backend_pdf import PdfPages

with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data_fx = pickle.load(fname)
with open(data_path+"ir.p", mode="rb") as fname:
    data_ir = pickle.load(fname)
with open(data_path+"events.p", mode="rb") as fname:
    events = pickle.load(fname)
with open(data_path+"data_dev_d.p", mode="rb") as fname:
    stocks = pickle.load(fname)["msci_ret"]


# Set start of the sample
start = "1997-04"  # every interest rate except for eur is available since then

fd = data_fx["fwd_disc"][start:]
sd = data_fx["spot_ret"][start:]
data_ir = data_ir[start:]
ir = data_ir


#stocks = stocks[start:]

# Get explicit interest rate differentials
ir_differentials = data_ir[start:].copy()
for col in ir_differentials.columns:
    ir_differentials[col] = ir_differentials[col]-ir_differentials["usd"]

ir_diff = ir_differentials.drop(["usd"], axis=1)

# .rolling(3).mean()
evts = events["fomc"]["1997-05":]
#with PdfPages(data_path+"ir_events/"+'boe_ir.pdf') as pdf:
for direction in ["all", "ups", "downs"]:
    es = event_study_wrapper(sd.mean(axis=1),
                             evts,
                             direction=direction,
                             window=[-12, -1, 0, 12],
                             ci_method="simple")
        #fig = es.plot()
        #plt.title(direction)
        #pdf.savefig(fig)


hikes = evts.where(evts.diff()>0, 0)
hikes = hikes.where(hikes==0, 1)

cuts = evts.where(evts.diff()<0, 0)
cuts = cuts.where(cuts==0, 1)

lookback = np.arange(5, 6, 1)
out = pd.DataFrame(columns=lookback)
for lb in lookback:
    rets = data_fx["spot_ret"].mean(axis=1).shift(0).rolling(lb).sum()

    retsh = rets.loc[hikes.where(hikes>0).dropna().index]
    retsh.cumsum().plot()

    retsc = rets.loc[cuts.where(cuts>0).dropna().index]
    retsc.cumsum().plot()

    conc = pd.concat([-1*retsh, retsc], axis=1).mean(axis=1)
    out[lb] = conc
    conc.cumsum().plot()


runup = ir.usd.diff().rolling(10).mean().shift(7)
runup2 = ir.usd.diff().rolling(66).mean().shift(7)
runup = runup.loc[hikes.index]
runup2 = runup2.loc[hikes.index]
cc = pd.concat([hikes, runup, runup2, runup-runup2], axis=1)
cc.columns = ['h', "s", 'l', 'd']

r = data_fx["spot_ret"].mean(axis=1).shift(1).rolling(5).sum()
rb = r.loc[cc.where((cc.d>0) & (cc.l>0)).dropna().index]
rs = r.loc[cc.where((cc.d<0) & (cc.l<0)).dropna().index]
combo = pd.concat([-rb, rs], axis=1).sum(axis=1).to_frame()
combo.cumsum().plot()

from foolbox import econometrics as ec

ols = ec.rOls(hikes, runup, True)

import statsmodels.api as sm
X = sm.add_constant(runup)
model = sm.Logit(hikes, X)

est  = model.fit()
model.predict(est.params)
est.params






















