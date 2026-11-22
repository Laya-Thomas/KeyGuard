
/*
    KeyGuard - CMU Keystroke Event Extractor

    Captures physical key events and timestamps.
    Does not store typed characters.

    Supports:
    - Period
    - Letters
    - Number 5
    - Shift
    - Enter
*/

class KeystrokeExtractor {

    constructor() {
        this.activeKeys = new Map();
        this.events = [];
        this.lastKeyDownTime = null;
        this.lastKeyUpTime = null;
        this.isCapturing = false;

        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleKeyUp = this.handleKeyUp.bind(this);
    }

    start() {
        this.reset();
        this.isCapturing = true;

        document.addEventListener(
            "keydown",
            this.handleKeyDown
        );

        document.addEventListener(
            "keyup",
            this.handleKeyUp
        );
    }

    handleKeyDown(event) {

        if (!this.isCapturing || event.repeat) {
            return;
        }

        // Ignore shortcuts involving Ctrl, Alt or Meta
        if (event.ctrlKey || event.altKey || event.metaKey) {
            return;
        }

        const now = performance.now();
        const keyCode = event.code;

        if (this.activeKeys.has(keyCode)) {
            return;
        }

        let dd = null;
        let ud = null;

        if (this.lastKeyDownTime !== null) {
            dd = now - this.lastKeyDownTime;
        }

        if (this.lastKeyUpTime !== null) {
            ud = now - this.lastKeyUpTime;
        }

        const record = {
            code: keyCode,
            down: now,
            up: null,
            hold: null,
            dd: dd,
            ud: ud
        };

        this.activeKeys.set(keyCode, record);
        this.events.push(record);

        this.lastKeyDownTime = now;
    }

    handleKeyUp(event) {

        if (!this.isCapturing) {
            return;
        }

        const keyCode = event.code;
        const record = this.activeKeys.get(keyCode);

        if (!record) {
            return;
        }

        const now = performance.now();

        record.up = now;
        record.hold = now - record.down;

        this.activeKeys.delete(keyCode);

        this.lastKeyUpTime = now;
    }

    stop() {

        this.isCapturing = false;

        document.removeEventListener(
            "keydown",
            this.handleKeyDown
        );

        document.removeEventListener(
            "keyup",
            this.handleKeyUp
        );

        return this.getFeatures();
    }

    getFeatures() {

        const completedEvents = this.events.filter(
            event => event.hold !== null
        );

        return {
            total_keystrokes: completedEvents.length,

            events: completedEvents.map(event => ({
                code: event.code,
                down: Number(event.down.toFixed(2)),
                up: Number(event.up.toFixed(2)),
                hold: Number(event.hold.toFixed(2)),
                dd: event.dd === null
                    ? null
                    : Number(event.dd.toFixed(2)),
                ud: event.ud === null
                    ? null
                    : Number(event.ud.toFixed(2))
            }))
        };
    }

    reset() {
        this.activeKeys.clear();
        this.events = [];
        this.lastKeyDownTime = null;
        this.lastKeyUpTime = null;
    }
}

window.KeystrokeExtractor = KeystrokeExtractor;
