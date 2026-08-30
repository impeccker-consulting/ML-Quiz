
from __future__ import annotations
import hashlib, json, random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOPIC_TARGETS={"Classification":7,"Regression":7,"Clustering":6}
DIFFICULTY_TARGETS={"Easy":6,"Moderate":12,"Difficult":2}
QUESTION_COUNT=20

def utc_now():
    return datetime.now(timezone.utc)

def iso_now():
    return utc_now().isoformat()

def parse_iso(value):
    if not value: return None
    return datetime.fromisoformat(value.replace("Z","+00:00"))

def deterministic_rng(sap_id, quiz_id):
    key=f"{str(sap_id).strip().upper()}|{str(quiz_id).strip().upper()}".encode()
    seed=int.from_bytes(hashlib.sha256(key).digest()[:16],"big")
    return random.Random(seed)

def load_bank(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data,dict) else data

def validate_bank(qs):
    ids=set()
    for q in qs:
        for k in ["Question_ID","Question_Family_ID","Topic","Business_Domain","Difficulty",
                  "Response_Type","Question_Type","Visual_Type","Question_Text","Options",
                  "Correct_Response","Marks","Variant_Enabled"]:
            if k not in q: raise ValueError(f"{q.get('Question_ID','?')}: missing {k}")
        if q["Question_ID"] in ids: raise ValueError(f"Duplicate Question_ID {q['Question_ID']}")
        ids.add(q["Question_ID"])
        if q["Topic"] not in TOPIC_TARGETS: raise ValueError(f"Invalid topic: {q['Topic']}")
        if q["Difficulty"] not in DIFFICULTY_TARGETS: raise ValueError(f"Invalid difficulty: {q['Difficulty']}")
        if q["Marks"] != 1: raise ValueError(f"{q['Question_ID']}: Marks must be 1")
        if q["Response_Type"] not in ("Single","Multi"): raise ValueError(f"{q['Question_ID']}: invalid response type")
        option_ids={o["Option_ID"] for o in q["Options"]}
        correct=set(q["Correct_Response"])
        if not correct or not correct <= option_ids: raise ValueError(f"{q['Question_ID']}: invalid answer key")
        if q["Response_Type"]=="Single" and len(correct)!=1: raise ValueError(f"{q['Question_ID']}: single must have one key")
        if q["Response_Type"]=="Multi" and len(correct)<2: raise ValueError(f"{q['Question_ID']}: multi must have 2+ keys")

def _allocation_choices(topic, available, rng):
    target=TOPIC_TARGETS[topic]
    out=[]
    for e in range(target+1):
        for m in range(target-e+1):
            d=target-e-m
            if d>1: continue
            if e<=available[(topic,"Easy")] and m<=available[(topic,"Moderate")] and d<=available[(topic,"Difficult")]:
                out.append({"Easy":e,"Moderate":m,"Difficult":d})
    rng.shuffle(out)
    return out

def _allocations(qs,rng):
    avail=Counter((q["Topic"],q["Difficulty"]) for q in qs)
    topics=list(TOPIC_TARGETS)
    choices={t:_allocation_choices(t,avail,rng) for t in topics}
    found=[]
    def bt(i,remaining,chosen):
        if i==len(topics):
            if all(v==0 for v in remaining.values()) and sum(chosen[t]["Difficult"]>0 for t in topics)>=2:
                found.append({(t,d):chosen[t][d] for t in topics for d in DIFFICULTY_TARGETS})
            return
        t=topics[i]
        for row in choices[t]:
            if any(row[d]>remaining[d] for d in DIFFICULTY_TARGETS): continue
            chosen[t]=row
            bt(i+1,{d:remaining[d]-row[d] for d in DIFFICULTY_TARGETS},chosen)
            del chosen[t]
    bt(0,dict(DIFFICULTY_TARGETS),{})
    rng.shuffle(found)
    return found

def select_questions(qs,sap_id,quiz_id):
    validate_bank(qs)
    rng=deterministic_rng(sap_id,quiz_id)
    allocations=_allocations(qs,rng)
    if not allocations: raise ValueError("No valid allocation satisfies frozen constraints.")
    allocation=allocations[0]
    pools={}
    for t in TOPIC_TARGETS:
        for d in DIFFICULTY_TARGETS:
            pool=[q for q in qs if q["Topic"]==t and q["Difficulty"]==d]
            rng.shuffle(pool); pools[(t,d)]=pool
    selected=[]
    for key,n in allocation.items(): selected += pools[key][:n]
    rng.shuffle(selected)
    result=[]
    for q in selected:
        x=dict(q); x["Options"]=list(q["Options"]); rng.shuffle(x["Options"]); result.append(x)
    return result

def attempt_path(root,quiz_id,sap_id):
    digest=hashlib.sha256(f"{sap_id.strip().upper()}|{quiz_id.strip().upper()}".encode()).hexdigest()[:24]
    return Path(root)/"attempts"/quiz_id/(digest+".json")

def load_attempt(root,quiz_id,sap_id):
    p=attempt_path(root,quiz_id,sap_id)
    if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    return None

def save_attempt(root,attempt):
    p=attempt_path(root,attempt["Quiz_ID"],attempt["SAP_ID"])
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(attempt,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(p)

def create_attempt(root,bank,config,sap_id):
    quiz_id=config["Quiz_ID"]
    existing=load_attempt(root,quiz_id,sap_id)
    if existing:
        return existing
    now=utc_now()
    start=parse_iso(config.get("Start_DateTime"))
    end=parse_iso(config.get("End_DateTime"))
    if start is not None and now < start:
        raise ValueError("Quiz has not started yet.")
    if end is not None and now > end:
        raise ValueError("Quiz is closed.")
    duration=int(config["Duration_Minutes"])
    deadline=now.timestamp()+duration*60
    if end is not None: deadline=min(deadline,end.timestamp())
    qs=select_questions(bank,sap_id,quiz_id)
    attempt={
      "SAP_ID":sap_id.strip().upper(),"Quiz_ID":quiz_id,
      "Started_Timestamp":now.isoformat(),"Deadline_Timestamp":datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
      "Submitted_Timestamp":None,"Submitted":False,"Auto_Submitted":False,"Score":None,
      "Questions":[
        {"Question_ID":q["Question_ID"],"Question_Family_ID":q["Question_Family_ID"],"Variant_ID":q.get("Variant_ID","V01"),
         "Topic":q["Topic"],"Difficulty":q["Difficulty"],"Question_Type":q["Question_Type"],
         "Response_Type":q["Response_Type"],"Question_Text":q["Question_Text"],"Options":q["Options"],
         "Marks":q["Marks"],"Correct_Response":q["Correct_Response"],
         "Shown_Timestamp":None,"First_Response_Timestamp":None,"Final_Response_Timestamp":None,
         "Time_Spent_Seconds":None,"Answer_Changed":False,"Final_Response":None,"Correct":None,"Answered":False}
        for q in qs
      ]
    }
    save_attempt(root,attempt); return attempt

def is_expired(attempt):
    return utc_now() >= parse_iso(attempt["Deadline_Timestamp"])

def score_attempt(attempt):
    total=0
    submitted_at=parse_iso(attempt.get("Submitted_Timestamp")) or utc_now()
    for q in attempt["Questions"]:
        response=q.get("Final_Response")
        if response is None or response==[] or response=="":
            q["Answered"]=False; q["Correct"]=False
            shown=parse_iso(q.get("Shown_Timestamp"))
            if shown is not None:
                q["Time_Spent_Seconds"]=max(0, int((submitted_at-shown).total_seconds()))
            continue
        submitted=set(response if isinstance(response,list) else [response])
        correct=set(q["Correct_Response"])
        q["Answered"]=True; q["Correct"]=submitted==correct
        shown=parse_iso(q.get("Shown_Timestamp"))
        if shown is not None:
            q["Time_Spent_Seconds"]=max(0, int((submitted_at-shown).total_seconds()))
        if q["Correct"]: total += q["Marks"]
    attempt["Score"]=total
    return total

def submit_attempt(root,attempt,auto=False):
    if attempt.get("Submitted"): return attempt
    now=utc_now()
    attempt["Submitted_Timestamp"]=now.isoformat()
    attempt["Submitted"]=True
    attempt["Auto_Submitted"]=bool(auto)
    score_attempt(attempt)
    save_attempt(root,attempt)
    return attempt
