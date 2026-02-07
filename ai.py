import asyncio
import sys
import os
import traceback
import pyaudio
import warnings
import datetime
import re
from google import genai
from google.genai import types

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
# 1. GOOGLE GEMINI API KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. SETTINGS
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
SILENCE_THRESHOLD = 600  # Adjusted for reliability
MIN_SPEECH_FRAMES = 3    # require X frames of sound to trigger "Speech" (debounce)
MIN_SILENCE_FRAMES = 10  # require X frames of silence to trigger "Silence"
VERBOSE_LOGGING = True   # Toggle detailed logs
# =================================================

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    GREY = '\033[90m'

def log(category, message):
    """Prints a timestamped log message if verbose logging is on."""
    if VERBOSE_LOGGING:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Use carriage return to overwrite the mic bar line if needed, then print new line
        # But since we want a scrolling log, we just print. 
        # Ideally we clear the mic line first if it was active.
        sys.stdout.write("\033[2K\r") # Clear current line
        print(f"{Colors.GREY}[{timestamp}]{Colors.RESET} [{category}] {message}")

class AudioAgent:
    def __init__(self, session_file):
        self.audio = pyaudio.PyAudio()
        self.send_queue = asyncio.Queue() 
        self.receive_queue = asyncio.Queue()
        self.running = True
        self.session_file = session_file
        
        # State tracking
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.user_turn_logged = False

    def is_junk_text(self, text):
        """Filters out non-conversational text."""
        text = text.strip()
        if not text: return True
        # Markdown/Tags
        if text.startswith(("*", "(", "[", "<", "#")): return True
        # Internal monologue indicators
        bad_phrases = [
            "i've", "i have", "i'm", "i am", "my plan", "my focus", "context:", 
            "identifying", "analyzing", "formulating", "registered", "instruction"
        ]
        lower = text.lower()
        # If it starts with a self-reference AND contains a process word
        if lower.startswith(("i ", "my ")):
            if any(p in lower for p in bad_phrases):
                return True
        return False

    def log_interaction(self, role, text):
        """Logs to file."""
        clean_text = re.sub(r'^\*\*.*?\*\*\s*', '', text).strip()
        clean_text = re.sub(r'^\(.*?\)\s*', '', clean_text).strip()
        
        if role == "Gemini" and self.is_junk_text(clean_text):
            log("FILTER", f"Filtered junk text: {clean_text[:50]}...")
            return

        try:
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(f"{role}: {clean_text}\n")
            log("FILE", f"Logged {role} interaction ({len(clean_text)} chars)")
        except Exception as e: 
            log("ERROR", f"File write error: {e}")

    def get_context_history(self):
        """Returns clean conversation history."""
        log("SYSTEM", "Loading context history...")
        if not os.path.exists(self.session_file): 
            log("SYSTEM", "No history file found.")
            return ""
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            clean_lines = []
            for line in lines:
                if line.startswith("User:") or line.startswith("Gemini:"):
                    # Double check filter on read to clean up old bad logs
                    content = line.split(":", 1)[1].strip()
                    if not self.is_junk_text(content):
                        clean_lines.append(line.strip())
            
            log("SYSTEM", f"Loaded {len(clean_lines)} lines of context.")
            return "\n".join(clean_lines[-15:]) # Context window
        except Exception as e:
            log("ERROR", f"Context load error: {e}")
            return ""

    async def run_microphone_task(self):
        CHUNK = 512
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNK)
        log("AUDIO", f"Microphone stream started. Chunk: {CHUNK}, Rate: 16000")
        print(f"{Colors.GREEN}🎤 Mic Ready.{Colors.RESET}")
        loop = asyncio.get_running_loop()
        
        while self.running:
            try:
                data = await loop.run_in_executor(None, lambda: stream.read(CHUNK, exception_on_overflow=False))
                
                # --- VAD LOGIC ---
                peak = 0
                if len(data) > 0:
                    # Simple peak amplitude
                    for i in range(0, len(data), 32):
                        val = abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True))
                        if val > peak: peak = val
                
                if peak > SILENCE_THRESHOLD:
                    self.speech_frames += 1
                    self.silence_frames = 0
                    if self.speech_frames == MIN_SPEECH_FRAMES: # Trigger ONCE
                        log("VAD", f"Speech Start detected (Peak: {peak})")
                    
                    if self.speech_frames >= MIN_SPEECH_FRAMES:
                        self.is_speaking = True
                        if not self.user_turn_logged:
                            sys.stdout.write(f"\r{Colors.GREEN}User: [Speaking...]{Colors.RESET}   ")
                            sys.stdout.flush()
                            self.user_turn_logged = True
                else:
                    self.silence_frames += 1
                    if self.silence_frames == MIN_SILENCE_FRAMES and self.is_speaking: # Trigger ONCE
                         log("VAD", "Speech End detected (Silence)")

                    self.speech_frames = 0
                    if self.silence_frames >= MIN_SILENCE_FRAMES:
                        self.is_speaking = False
                        self.user_turn_logged = False # Reset for next turn

                # Always send data to keep connection alive
                self.send_queue.put_nowait(data)

            except Exception as e:
                log("ERROR", f"Mic read error: {e}")
        
        stream.stop_stream()
        stream.close()
        log("AUDIO", "Microphone stream closed")

    async def send_to_gemini(self, session):
        log("NET", "Audio sender task started")
        while self.running:
            try:
                chunk = await self.send_queue.get()
                # Use standard 'input' argument for bytes
                await session.send(input={"data": chunk, "mime_type": "audio/pcm"}, end_of_turn=False)
            except asyncio.CancelledError: 
                log("NET", "Sender task cancelled")
                break
            except Exception as e: 
                log("NET", f"Send error: {e}")
                await asyncio.sleep(0.1)

    async def play_audio(self):
        log("AUDIO", "Output stream started")
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        while self.running:
            try:
                data = await self.receive_queue.get()
                await asyncio.to_thread(stream.write, data)
            except asyncio.CancelledError: 
                log("AUDIO", "Player task cancelled")
                break
            except Exception as e: 
                log("ERROR", f"Playback error: {e}")
        stream.stop_stream()
        stream.close()
        log("AUDIO", "Output stream closed")

    def cleanup(self):
        log("SYSTEM", "Cleaning up resources...")
        self.running = False
        self.audio.terminate()

async def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Colors.CYAN}--- GEMINI LIVE AGENT v12.2 (Clean Logs) ---{Colors.RESET}")

    if not GEMINI_API_KEY:
        print(f"{Colors.RED}Error: GEMINI_API_KEY not found.{Colors.RESET}")
        return

    if not os.path.exists("context"): os.makedirs("context")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_filename = os.path.join("context", f"session_{timestamp}.txt")
    with open(session_filename, "w", encoding="utf-8") as f:
        f.write(f"--- Session {timestamp} ---\n")
    log("SYSTEM", f"Session file created: {session_filename}")

    agent = AudioAgent(session_filename)
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
    
    # SYSTEM PROMPT INJECTED WITH HISTORY
    # We define a function to get config so we can refresh history on reconnect
    def get_config():
        history = agent.get_context_history()
        # Prompt Engineering: Force 'Role' based behavior to stop monologue
        system_msg = """You are 'Puck', a helpful voice assistant.
        RULES:
        1. YOU ARE HAVING A VOICE CONVERSATION. SPEAK DIRECTLY TO THE USER.
        2. NEVER DESCRIBE YOUR ACTIONS OR THOUGHT PROCESS.
        3. DO NOT OUTPUT MARKDOWN HEADERS LIKE **Thinking** OR **Response**.
        4. BE CONCISE AND CONVERSATIONAL."""
        
        if history:
            log("SYSTEM", "Injecting context history into system instruction.")
            system_msg += f"\n\nPREVIOUS TRANSCRIPT:\n{history}\n\n[SYSTEM]: Connection restored. The user just spoke. Respond to them naturally."
        
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=system_msg)])
        )

    mic_task = asyncio.create_task(agent.run_microphone_task())
    speaker_task = asyncio.create_task(agent.play_audio())

    first_run = True

    while True:
        try:
            print(f"\n{Colors.YELLOW}Connecting...{Colors.RESET}")
            log("NET", f"Attempting connection to {MODEL_ID}")
            
            async with client.aio.live.connect(model=MODEL_ID, config=get_config()) as session:
                log("NET", "WebSocket connection established")
                print(f"{Colors.GREEN}✅ Connected.{Colors.RESET}")
                
                if first_run:
                    print(f"{Colors.GREEN}👉 Speak now...{Colors.RESET}")
                    first_run = False
                
                send_task = asyncio.create_task(agent.send_to_gemini(session))
                first_text = True

                async for message in session.receive():
                    if message.server_content is None: 
                        log("NET", "Received empty server content")
                        continue
                    
                    model_turn = message.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.text:
                                text = part.text.strip()
                                if not text: continue
                                
                                # Strict Filter
                                if agent.is_junk_text(text): 
                                    log("FILTER", f"Ignored junk text: {text[:30]}...")
                                    continue
                                
                                # Log User Turn Marker if this is the start of a response
                                if first_text:
                                    agent.log_interaction("User", "[User Spoke]")
                                    first_text = False
                                    sys.stdout.write("\033[2K\r")
                                    print(f"{Colors.CYAN}Gemini: {Colors.RESET}", end="", flush=True)
                                
                                agent.log_interaction("Gemini", text)
                                print(text, end="", flush=True)
                            
                            if part.inline_data:
                                await agent.receive_queue.put(part.inline_data.data)

                    if message.server_content.interrupted:
                        log("NET", "Received INTERRUPTION signal from server")
                        sys.stdout.write("\r" + " " * 50 + "\r")
                        print(f"\n{Colors.RED}[Interrupted]{Colors.RESET}")
                        # Clear audio output buffer so AI stops talking instantly
                        cleared_count = 0
                        while not agent.receive_queue.empty():
                            try: 
                                agent.receive_queue.get_nowait()
                                cleared_count += 1
                            except: break
                        log("AUDIO", f"Cleared {cleared_count} chunks from output queue")
                        first_text = True

                    if message.server_content.turn_complete:
                        log("NET", "Received Turn Complete signal")
                        print("")
                        first_text = True

                log("NET", "Server closed the connection (iteration end)")
                print(f"\n{Colors.YELLOW}Disconnected.{Colors.RESET}")

        except asyncio.CancelledError:
            log("SYSTEM", "Main loop cancelled")
            break
        except Exception as e:
            log("ERROR", f"Connection/Session error: {e}")
            print(f"\n{Colors.RED}Reconnecting... {e}{Colors.RESET}")
            await asyncio.sleep(1)
        
        try: 
            send_task.cancel()
            log("NET", "Sender task cleanup initiated") 
        except: pass

    agent.cleanup()

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass