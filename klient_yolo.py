import cv2, pygame, io, threading, time
from ultralytics import YOLO
from gtts import gTTS

model = YOLO("best.pt")
w,h=640,480
cap=cv2.VideoCapture(0)
cap.set(3,w); cap.set(4,h)
pygame.init(); screen=pygame.display.set_mode((w,h+50)); font=pygame.font.SysFont(None,16)
buttons={"camera":(10,h+10,150,30),"yolo":(170,h+10,150,30),"snapshot":(330,h+10,150,30)}
camera_on, yolo_live_on, paused = True, False, False
frozen_frame, current_frame, last_labels = None, None, []

tts_queue=[]; tts_lock=threading.Lock()
def speak(t):
    with tts_lock: tts_queue.append(t)
def tts_worker():
    while True:
        if tts_queue:
            t=tts_queue.pop(0)
            tts=gTTS(text=t,lang='pl'); fp=io.BytesIO(); tts.write_to_fp(fp); fp.seek(0)
            pygame.mixer.init(); pygame.mixer.music.load(fp,'mp3'); pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
        else: time.sleep(0.05)
threading.Thread(target=tts_worker,daemon=True).start()

def draw_buttons():
    for _,(x,y,w_,h_) in buttons.items(): pygame.draw.rect(screen,(180,180,180),(x,y,w_,h_))
    screen.blit(font.render("Kamera on/off",1,(0,0,0)),(15,h+15))
    screen.blit(font.render("yolo on/off",1,(0,0,0)),(175,h+15))
    screen.blit(font.render("pseudo-zdjęcie + yolo",1,(0,0,0)),(335,h+15))

running=True
while running:
    for e in pygame.event.get():
        if e.type==pygame.QUIT: running=False
        elif e.type==pygame.MOUSEBUTTONDOWN:
            mx,my=pygame.mouse.get_pos()
            for name,(x,y,w_,h_) in buttons.items():
                if x<=mx<=x+w_ and y<=my<=y+h_:
                    if name=="camera": camera_on=not camera_on; speak("Kamera włączona" if camera_on else "Kamera wyłączona")
                    elif name=="yolo": yolo_live_on=not yolo_live_on; speak("YOLO włączone" if yolo_live_on else "YOLO wyłączone")
                    elif name=="snapshot": paused=not paused; frozen_frame=current_frame.copy() if paused and current_frame is not None else frozen_frame; speak("Stopklatka" if paused else "Wideo na żywo")
    if camera_on:
        if not paused:
            ret,frame=cap.read()
            if ret: frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); current_frame=frame.copy(); display_frame=frame.copy()
            if yolo_live_on and ret:
                results=model(display_frame)[0]; display_frame=results.plot()
                current_labels=[results.names[int(b.cls)] for b in results.boxes]
                for l in current_labels:
                    if l not in last_labels: speak(f"Wykryto {l}")
                last_labels=current_labels
        else:
            display_frame=frozen_frame.copy() if frozen_frame is not None else current_frame
            if yolo_live_on and display_frame is not None:
                results=model(display_frame)[0]; display_frame=results.plot()
        if display_frame is not None: screen.blit(pygame.surfarray.make_surface(display_frame.swapaxes(0,1)),(0,0))
    draw_buttons(); pygame.display.update()

pygame.quit(); cap.release()
