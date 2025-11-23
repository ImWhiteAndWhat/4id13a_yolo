import cv2, pygame, io, threading, time
from ultralytics import YOLO
from gtts import gTTS

CONF_THRESHOLD = 0.5
TTS_MAX_AGE = 3
TTS_MAX_QUEUE_LENGTH = 5

model = YOLO("best.pt")
w, h = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, w)
cap.set(4, h)

pygame.init()
screen = pygame.display.set_mode((w, h + 50))
font = pygame.font.SysFont(None, 16)

b = {
    "camera": (10, h + 10, 150, 30),
    "yolo": (170, h + 10, 150, 30),
    "snapshot": (330, h + 10, 150, 30)
}

cam_on = True
yolo_on = False
paused = False

frozen = None
cur = None
last = []

q = []
ql = threading.Lock()

def speak(t):
    with ql:q.append((time.time(),t));len(q)>TTS_MAX_QUEUE_LENGTH and q.pop(0)

def tts_worker():
    while 1:
        with ql:(ts,tx)=q.pop(0) if q else (None,None)
        if not ts:time.sleep(.05);continue
        if time.time()-ts>TTS_MAX_AGE:continue
        fp=io.BytesIO();gTTS(text=tx,lang='pl').write_to_fp(fp);fp.seek(0)
        pygame.mixer.init();pygame.mixer.music.load(fp,'mp3');pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():time.sleep(.1)

threading.Thread(target=tts_worker,daemon=1).start()

def draw_btns():
    for x in b.values():pygame.draw.rect(screen,(180,180,180),x)
    screen.blit(font.render("Kamera on/off",1,(0,0,0)),(15,h+15))
    screen.blit(font.render("yolo on/off",1,(0,0,0)),(175,h+15))
    screen.blit(font.render("pseudo-zdjęcie + yolo",1,(0,0,0)),(335,h+15))

run=1
while run:
    for e in pygame.event.get():
        if e.type==pygame.QUIT:run=0
        elif e.type==pygame.MOUSEBUTTONDOWN:
            mx,my=pygame.mouse.get_pos()
            for k,(x,y,W,H) in b.items():
                if x<=mx<=x+W and y<=my<=y+H:
                    if k=="camera":cam_on=not cam_on;speak("Kamera włączona"if cam_on else"Kamera wyłączona")
                    elif k=="yolo":yolo_on=not yolo_on;speak("YOLO włączone"if yolo_on else"YOLO wyłączone")
                    else:paused=not paused;frozen=cur.copy()if paused and cur is not None else frozen;speak("Stopklatka"if paused else"Wideo na żywo")
    if cam_on:
        if not paused:
            r,f=cap.read();
            if r:f=cv2.cvtColor(f,cv2.COLOR_BGR2RGB);cur=f.copy();d=f.copy()
            if yolo_on and r:
                res=model(d,conf=CONF_THRESHOLD)[0];d=res.plot();cl=[res.names[int(b.cls)]for b in res.boxes if b.conf>=CONF_THRESHOLD]
                [speak(f"Wykryto {l}") for l in cl if l not in last];last=cl
        else:
            d=frozen.copy()if frozen is not None else cur
            if yolo_on and d is not None:d=model(d,conf=CONF_THRESHOLD)[0].plot()
        if d is not None:screen.blit(pygame.surfarray.make_surface(d.swapaxes(0,1)),(0,0))
    draw_btns();pygame.display.update()
pygame.quit();cap.release()
