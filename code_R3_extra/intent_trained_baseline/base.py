import csv,sys,numpy as np,collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from scipy.stats import binomtest
P='../ip_res/zip/r3_intent_parser_package'
C=list(csv.DictReader(open(P+'/intent_corpus.csv')))
excl={'Q067','Q107'}
err={e['id'] for e in csv.DictReader(open(P+'/analysis/errors.csv'))}
kw={r['id']:int(r['keyword_ok']) for r in csv.DictReader(open(P+'/baseline_results.csv'))}
labs=lambda g: frozenset() if g=='none' else frozenset(g.split(';'))
ATOM=sorted({a for r in C for a in labs(r['gold'])})
texts=[r['text'] for r in C]; Y=[labs(r['gold']) for r in C]
def vec(): return make_union(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),sublinear_tf=True),TfidfVectorizer(analyzer='word',ngram_range=(1,2),sublinear_tf=True))
def loo(method):
    pred=[]
    for i in range(len(C)):
        tr=[j for j in range(len(C)) if j!=i]
        v=vec(); Xtr=v.fit_transform([texts[j] for j in tr]); xi=v.transform([texts[i]])
        if method=='1nn':
            sims=(Xtr@xi.T).toarray().ravel(); pred.append(Y[tr[int(np.argmax(sims))]])
        else:
            s=set()
            for a in ATOM:
                y=np.array([a in Y[j] for j in tr])
                if y.sum()==0: continue
                m=LogisticRegression(C=10,max_iter=2000,class_weight='balanced').fit(Xtr,y)
                if m.predict_proba(xi)[0,1]>0.5: s.add(a)
            pred.append(frozenset(s))
    return pred
res={}
for meth in ['1nn','logreg']:
    pr=loo(meth)
    ok={C[i]['id']:int(pr[i]==Y[i]) for i in range(len(C))}
    res[meth]=ok
ids=[r['id'] for r in C if r['id'] not in excl]
lev={r['id']:r['level'] for r in C}
parser={i:int(i not in err) for i in ids}
print('total 109: parser',sum(parser[i] for i in ids),'keyword',sum(kw[i] for i in ids),*[(m,sum(res[m][i] for i in ids)) for m in res])
for L in ['L0','L1','L2','L3','L4','L5','RU','C','H','N','O']:
    s=[i for i in ids if lev[i]==L]
    print(L,len(s),'parser',sum(parser[i] for i in s),'kw',sum(kw[i] for i in s),*[(m,sum(res[m][i] for i in s)) for m in res])
for m in res:
    b=sum(1 for i in ids if parser[i] and not res[m][i]); c=sum(1 for i in ids if res[m][i] and not parser[i])
    print(m,'McNemar parser-only',b,'clf-only',c,'p',binomtest(b,b+c,0.5).pvalue if b+c else 1)
    s=[i for i in ids if lev[i]=='L5']
    b=sum(1 for i in s if parser[i] and not res[m][i]); c=sum(1 for i in s if res[m][i] and not parser[i])
    print(' L5',b,c,binomtest(b,b+c,0.5).pvalue if b+c else 1)
import json; json.dump(res,open('base_results.json','w'))
