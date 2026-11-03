"""TwiML document generation.

Builds the XML Twilio executes when an outbound call is answered. For the
media-stream flow, the TwiML connects the call to our WebSocket endpoint:

.. code-block:: xml

    <Response>
      <Connect>
        <Stream url="wss://host/ws/media-stream/CAxxxx" />
      </Connect>
    </Response>
"""

from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring


def build_stream_twiml(stream_ws_url: str) -> str:
    """TwiML that connects the answered call to a Media Stream WebSocket."""
    response = Element("Response")
    connect = SubElement(response, "Connect")
    SubElement(connect, "Stream", url=stream_ws_url)
    return tostring(response, encoding="unicode", xml_declaration=True)


def build_say_twiml(message: str, voice: str = "alice") -> str:
    """TwiML that speaks a message with TTS and then hangs up."""
    response = Element("Response")
    say = SubElement(response, "Say", voice=voice)
    say.text = message
    SubElement(response, "Hangup")
    return tostring(response, encoding="unicode", xml_declaration=True)


def build_reject_twiml(reason: str = "This call cannot be completed.") -> str:
    """TwiML used to reject calls (e.g., failed webhook validation fallback)."""
    response = Element("Response")
    say = SubElement(response, "Say", voice="alice")
    say.text = reason
    SubElement(response, "Hangup")
    return tostring(response, encoding="unicode", xml_declaration=True)


__all__ = ["build_stream_twiml", "build_say_twiml", "build_reject_twiml"]
