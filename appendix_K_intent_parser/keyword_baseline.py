"""Frozen keyword baseline for the Intent Parser test (written once, not tuned on results).
Detects the quantity by topic words and the direction by direction words in the same clause.
Output: set of intent codes, or {'none'}."""
import re,csv,sys
TOPIC={
 'mortality':r'mortalit|fatal|death|die|cfr|летальн|умерш|смерт',
 'growth':r'growth|grow|rise|increase|spread|doubling|steep|flat|start|early|initial|рост|растi|расти',
 'day':r'earl|soon|late|delay|before|after|date|when|time|forward|back|closer|further|wait|culminat|turning point|раньше|позже|поздн|дальше|дат',
 'height':r'high|low|large|big|small|modest|more|fewer|less|maximum|top|height|peak|exceed|overestimat|underestimat|inflat|cut|выше|ниже|больше|меньше|максимум|завыш|заниж',
}
UP=r'higher|larger|bigger|more\b|exceed|underestimat|too (low|small|modest|slow|flat|optimistic)|go up|faster|quick|steeper|shorter|speed up|larger proportion|выше|больше|быстрее|круче|заниж'
DOWN=r'lower|smaller|fewer|less\b|less steep|overestimat|inflated|cut|come down|too high|slower|ниже|меньше|завыш'
EARLY=r'earlier|sooner|before|ahead|forward|closer|too late|раньше'
LATE=r'later|delay|after|longer|further|back in time|too early|позже|поздн|дальше'
NEG=r"\b(don't|not|никак не|не)\b"
def parse(text):
    t=text.lower()
    out=set()
    clauses=re.split(r'[;,.]| and | but | so |, а | и ',t)
    for c in clauses:
        if not c.strip(): continue
        neg=bool(re.search(NEG,c))
        if re.search(TOPIC['mortality'],c):
            if re.search(UP,c): out.add('mortality_lower' if neg else 'mortality_higher')
            elif re.search(DOWN,c): out.add('mortality_higher' if neg else 'mortality_lower')
            continue
        if re.search(r'growth|grow|rise|spread|doubling|рост|расти',c) and re.search(r'start|early|initial|first|beginning|начальн|перв',c+' '+t):
            if re.search(r'slower|less steep',c): out.add('growth_faster' if neg else 'growth_slower')
            elif re.search(UP+r'|too slow|this slow',c): out.add('growth_faster')
            continue
        if re.search(r'starts off too flat',c): out.add('growth_faster'); continue
        if re.search(EARLY,c): out.add('peak_later' if neg else 'peak_earlier'); continue
        if re.search(LATE,c): out.add('peak_earlier' if neg else 'peak_later'); continue
        if re.search(r'peak|maximum|top|height|wave|curve|cases|patients|пик|максимум|больных|заболевш',c):
            if re.search(r'as low as|too (low|small|modest)',c): out.add('peak_higher'); continue
            if re.search(r'as high as|too high',c): out.add('peak_lower'); continue
            if re.search(UP,c): out.add('peak_lower' if neg else 'peak_higher')
            elif re.search(DOWN,c): out.add('peak_higher' if neg else 'peak_lower')
    return out or {'none'}
TEMPLATE={'peak should be higher':'peak_higher','peak should be lower':'peak_lower','peak should be earlier':'peak_earlier',
          'peak should be later':'peak_later','faster initial growth':'growth_faster','higher mortality':'mortality_higher'}
def template_rule(text):
    k=text.strip().lower().rstrip('.')
    return {TEMPLATE[k]} if k in TEMPLATE else {'none'}
if __name__=='__main__':
    rows=list(csv.DictReader(open(sys.argv[1],encoding='utf-8')))
    w=csv.writer(open(sys.argv[2],'w',newline='',encoding='utf-8'))
    w.writerow(['id','group','level','gold','template_rule','keyword_rule','template_ok','keyword_ok'])
    for r in rows:
        g=set(r['gold'].split(';')); a=template_rule(r['text']); b=parse(r['text'])
        w.writerow([r['id'],r['group'],r['level'],r['gold'],';'.join(sorted(a)),';'.join(sorted(b)),int(a==g),int(b==g)])
