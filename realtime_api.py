"""
realtime_api.py

This module implements a client for the OpenAI Realtime API using WebSocket. 
It handles audio input and output, allowing for real-time audio processing and 
communication with the API. The AudioHandler class manages audio playback, 
while the RealtimeAPIClient class manages the WebSocket connection and 
communication with the OpenAI API. The client captures audio from the 
microphone, sends it to the API, and plays back the audio responses in real-time.
"""

import base64
import json
import threading
import time
import os
import pyaudio
from websocket import WebSocketApp
import logging
from io import BytesIO

# ----------------------------- Logging Configuration -----------------------------
# Configure logging to display timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------- Configuration Constants -----------------------------
# OpenAI Realtime API WebSocket URL
WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Read OpenAI API Key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    exit(1)  # Exit if API key is not set

# ----------------------------- Audio Handling -----------------------------

class AudioHandler:
    def __init__(self, output_device_id=None):
        self.p = pyaudio.PyAudio()  # Initialize PyAudio
        self.output_device_id = output_device_id  # Set output device ID

        # Set up audio stream for playback
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,  # 16-bit audio format
                channels=1,  # Mono audio
                rate=24000,  # Must match the API's expected rate
                output=True,  # Enable output
                frames_per_buffer=1024,  # Buffer size for audio frames
                output_device_index=self.output_device_id  # Specify output device
            )
            logger.info("Audio playback stream opened successfully.")
        except Exception as e:
            logger.error(f"Failed to open audio playback stream: {e}")
            raise e  # Raise exception if stream fails to open

        self.audio_queue = []  # Queue to hold audio data
        self.lock = threading.Lock()  # Lock for thread-safe access to the queue
        self.stop_event = threading.Event()  # Event to signal stopping playback

        # Start playback thread
        self.playback_thread = threading.Thread(target=self.play_audio, daemon=True)
        self.playback_thread.start()  # Start the audio playback thread
        logger.info("Audio playback thread started.")

    def enqueue_audio(self, audio_data):
        """Add audio data to the playback queue."""
        with self.lock:
            self.audio_queue.append(audio_data)  # Append audio data to the queue
            logger.debug(f"Enqueued audio chunk of size {len(audio_data)} bytes. Queue size: {len(self.audio_queue)}")

    def play_audio(self):
        """Continuously play audio from the queue."""
        while not self.stop_event.is_set():  # Run until stop event is set
            with self.lock:
                if self.audio_queue:
                    audio_data = self.audio_queue.pop(0)  # Get the next audio chunk
                    logger.debug(f"Dequeued audio chunk of size {len(audio_data)} bytes.")
                else:
                    audio_data = None  # No audio data to play
            if audio_data:
                try:
                    # Play audio
                    self.stream.write(audio_data)  # Write audio data to the stream
                    logger.debug(f"Played audio chunk of size {len(audio_data)} bytes.")
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")  # Log any playback errors
            else:
                time.sleep(0.01)  # Sleep briefly to prevent CPU overuse

    def close(self):
        """Close the audio stream and stop playback."""
        self.stop_event.set()  # Signal to stop playback
        self.playback_thread.join()  # Wait for playback thread to finish
        self.stream.stop_stream()  # Stop the audio stream
        self.stream.close()  # Close the audio stream
        self.p.terminate()  # Terminate PyAudio
        logger.info("Audio playback streams closed.")

# ----------------------------- WebSocket Client -----------------------------

class RealtimeAPIClient:
    def __init__(self, api_key, websocket_url, audio_handler):
        self.api_key = api_key  # Store API key
        self.websocket_url = websocket_url  # Store WebSocket URL
        self.audio_handler = audio_handler  # Store audio handler instance

        # Initialize the WebSocketApp
        self.ws_app = WebSocketApp(
            self.websocket_url,
            header=[
                f"Authorization: Bearer {self.api_key}",  # Set authorization header
                "OpenAI-Beta: realtime=v1"  # Set OpenAI beta header
            ],
            on_open=self.on_open,  # Set callback for when connection opens
            on_message=self.on_message,  # Set callback for incoming messages
            on_error=self.on_error,  # Set callback for errors
            on_close=self.on_close  # Set callback for connection close
        )

        # Lock for sending messages
        self.send_lock = threading.Lock()  # Lock to ensure thread-safe sending of messages

        # Audio capture parameters
        self.frame_duration = 30  # Duration of each audio frame in milliseconds
        self.sample_rate = 24000  # Sample rate in Hz for API
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000) * 2  # Frame size in bytes (2 bytes per sample)

    def on_open(self, ws):
        """Callback for when the WebSocket connection is opened."""
        logger.info("WebSocket connection opened.")
        # Send session.update event to configure the session
        self.send_session_update()
        # Start a thread to continuously capture and send audio
        threading.Thread(target=self.continuous_audio_capture, daemon=True).start()

    def send_session_update(self):
        """
        Send the session.update event to configure session properties.
        """
        try:
            event = {
                "type": "session.update",  # Event type
                "session": {
                    "modalities": ["text", "audio"],  # Supported modalities
                    "instructions": "Your knowledge cutoff is 2023-10. Assist the user with their requests.",
                    "voice": "alloy",  # Voice model to use
                    "input_audio_format": "pcm16",  # Input audio format
                    "output_audio_format": "pcm16",  # Output audio format
                    "input_audio_transcription": {
                        "enabled": True,  # Enable audio transcription
                        "model": "whisper-1"  # Model for transcription
                    },
                    "turn_detection": {
                        "type": "server_vad",  # Type of turn detection
                        "threshold": 0.5,  # Threshold for voice activity detection
                        "prefix_padding_ms": 300,  # Padding before speech
                        "silence_duration_ms": 200  # Duration of silence to detect end of speech
                    },
                    "tools": [],  # Placeholder for any tools to be added
                    "tool_choice": "auto",  # Automatic tool choice
                    # Removed "temperature" and "max_output_tokens" from session.update
                }
            }
            with self.send_lock:
                self.ws_app.send(json.dumps(event))  # Send the session update event
                logger.info("Sent session.update event to configure session.")
        except Exception as e:
            logger.error(f"Error sending session.update event: {e}")  # Log any errors during sending

    def send_response_create(self):
        """
        Send the response.create event to initiate response generation.
        """
        try:
            event = {
                "type": "response.create",  # Event type
                "response": {
                    "modalities": ["text", "audio"],  # Supported modalities
                    "instructions": "Please assist the user."  # Instructions for the response
                }
            }
            with self.send_lock:
                self.ws_app.send(json.dumps(event))  # Send the response create event
                logger.info("Sent response.create event to initiate response generation.")
        except Exception as e:
            logger.error(f"Error sending response.create event: {e}")  # Log any errors during sending

    def continuous_audio_capture(self):
        """
        Continuously capture audio from the microphone and send it to the API.
        """
        p = pyaudio.PyAudio()  # Initialize PyAudio for audio input
        try:
            stream = p.open(
                format=pyaudio.paInt16,  # 16-bit audio format
                channels=1,  # Mono audio
                rate=self.sample_rate,  # Sample rate
                input=True,  # Enable input
                frames_per_buffer=self.frame_size // 2  # Number of frames per buffer
            )
            logger.info("Audio input stream opened successfully.")
        except Exception as e:
            logger.error(f"Failed to open audio input stream: {e}")  # Log any errors during stream opening
            return

        while not self.audio_handler.stop_event.is_set():  # Run until stop event is set
            try:
                # Read raw audio data
                audio_data = stream.read(self.frame_size // 2, exception_on_overflow=False)  # Read audio data
                if len(audio_data) < self.frame_size:  # Check for incomplete frames
                    logger.warning("Incomplete audio frame received.")
                    continue

                # Encode to base64
                base64_audio = base64.b64encode(audio_data).decode('utf-8')  # Encode audio data to base64

                # Create input_audio_buffer.append event
                event = {
                    "type": "input_audio_buffer.append",  # Event type
                    "audio": base64_audio  # Base64 encoded audio data
                }

                with self.send_lock:
                    self.ws_app.send(json.dumps(event))  # Send the audio buffer append event
                    logger.debug(f"Sent input_audio_buffer.append event with audio chunk of size {len(audio_data)} bytes.")

            except Exception as e:
                logger.error(f"Error during audio capture and sending: {e}")  # Log any errors during capture and sending
                break

        stream.stop_stream()  # Stop the audio input stream
        stream.close()  # Close the audio input stream
        p.terminate()  # Terminate PyAudio
        logger.info("Audio input stream closed.")

    def on_message(self, ws, message):
        """Callback for when a message is received from the WebSocket."""
        try:
            # Parse the incoming JSON message
            event = json.loads(message)  # Decode JSON message
            event_type = event.get('type', '')  # Get the event type
            logger.debug(f"Received event type: {event_type}")

            if event_type == 'response.audio.delta':
                delta = event.get('delta', '')  # Get audio delta
                if delta:
                    audio_bytes = base64.b64decode(delta)  # Decode base64 audio data
                    if audio_bytes:
                        self.audio_handler.enqueue_audio(audio_bytes)  # Enqueue audio for playback
                        logger.info(f"Received and enqueued audio chunk of size {len(audio_bytes)} bytes.")
                    else:
                        logger.warning("Received empty audio data in response.audio.delta.")
                else:
                    logger.warning("No delta found in response.audio.delta.")

            elif event_type == 'error':
                error = event.get('error', {})  # Get error details
                logger.error(f"Server Error: Type: {error.get('type', 'N/A')}, "
                             f"Code: {error.get('code', 'N/A')}, "
                             f"Message: {error.get('message', 'N/A')}, "
                             f"Param: {error.get('param', 'N/A')}, "
                             f"Event ID: {error.get('event_id', 'N/A')}")

            elif event_type == 'response.audio.done':
                logger.info("Received response.audio.done. Audio response completed.")

            elif event_type == 'response.text.delta':
                text_data = event.get('delta', '')  # Get text delta
                if text_data:
                    logger.info(f"Received text: {text_data}")

            elif event_type == 'response.audio_transcript.done':
                transcript = event.get('transcript', '')  # Get audio transcript
                logger.info(f"Audio Transcript: {transcript}")

            elif event_type == 'response.done':
                response = event.get('response', {})  # Get response details
                status = response.get('status', '')  # Get response status
                logger.info(f"Response Completed. Status: {status}")

            elif event_type == 'rate_limits.updated':
                rate_limits = event.get('rate_limits', [])  # Get rate limits
                for rl in rate_limits:
                    logger.info(f"Rate Limit Updated - Name: {rl.get('name')}, "
                                f"Limit: {rl.get('limit')}, Remaining: {rl.get('remaining')}, "
                                f"Reset Seconds: {rl.get('reset_seconds')}")

            else:
                logger.debug(f"Unhandled event type: {event_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e} - Message: {message}")  # Log JSON decode errors
        except Exception as e:
            logger.error(f"Unexpected error in on_message: {e} - Message: {message}")  # Log unexpected errors

    def on_error(self, ws, error):
        """
        Handle WebSocket errors.
        """
        logger.error(f"WebSocket error: {error}")  # Log WebSocket errors

    def on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket closure.
        """
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")  # Log closure details

    def run(self):
        """
        Run the WebSocketApp.
        """
        logger.info("Starting WebSocket client...")
        self.ws_app.run_forever()  # Start the WebSocket client

# ----------------------------- Main Execution -----------------------------

def main():
    # Initialize AudioHandler without specifying an output device (uses default)
    audio_handler = AudioHandler()

    # Initialize RealtimeAPIClient
    client = RealtimeAPIClient(
        api_key=OPENAI_API_KEY,
        websocket_url=WEBSOCKET_URL,
        audio_handler=audio_handler
    )

    try:
        # Run the client in a separate thread
        client_thread = threading.Thread(target=client.run, daemon=True)
        client_thread.start()  # Start the WebSocket client thread

        # Keep the main thread alive and allow for graceful shutdown
        while True:
            time.sleep(1)  # Sleep to keep the main thread alive
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        audio_handler.stop_event.set()  # Signal all threads to stop
        client.ws_app.close()  # Close the WebSocket connection
        client_thread.join(timeout=5)  # Wait for the client thread to finish
    except Exception as e:
        logger.error(f"Unexpected error: {e}")  # Log any unexpected errors
    finally:
        audio_handler.close()  # Close the audio handler

if __name__ == "__main__":
    main()  # Execute the main function
