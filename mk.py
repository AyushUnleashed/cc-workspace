import wave,numpy as np
from faster_whisper import WhisperModel
w=wave.open('a.wav');a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
s,_=WhisperModel('base.en',compute_type='int8').transcribe(a,word_timestamps=True)
W=[x for g in s for x in g.words]
def t(x):
    h=int(x//3600);m=int(x%3600//60);return f"{h}:{m:02d}:{x%60:05.2f}"
hdr="""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: C,Inter,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,7,2,8,60,60,150,1
[Events]
Format: Layer,Start,End,Style,Name,Text
"""
ev=[];i=0
while i<len(W):
    ch=W[i:i+3];i+=3
    txt=" ".join(x.word.strip() for x in ch).upper().strip(",.")
    st=ch[0].start;en=ch[-1].end
    ev.append((st,en,txt))
out=hdr
for k,(st,en,tx) in enumerate(ev):
    if k+1<len(ev): en=min(max(en,st+.3),ev[k+1][0]) if ev[k+1][0]>en else en
    out+=f"Dialogue: 0,{t(st)},{t(en+0.05)},C,,{tx}\n"
open('c.ass','w').write(out)
