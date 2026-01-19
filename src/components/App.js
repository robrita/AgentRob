import { icons } from "../utils/icons.js";

export const App = () => `
  <div class="app">
    <div
      id="loadingScreen"
      class="loading-overlay"
      role="status"
      aria-live="polite"
      aria-label="Connecting to backend"
    >
      <div class="loading-card">
        <div class="loading-spinner" aria-hidden="true"></div>
        <div class="loading-title">Connecting to AgentRob…</div>
        <div class="loading-subtitle">Warming up the backend, please wait.</div>
      </div>
    </div>
    <header class="topbar">
      <div class="topbar__title">
        <span class="brand">AgentRob</span>
        <span class="divider">|</span>
        <span class="subtitle">Conversation with AgentRob</span>
      </div>
      <div class="topbar__actions">
        <button class="icon-button" aria-label="Help" title="Help">
          <span class="icon" aria-hidden="true">${icons.help}</span>
        </button>
        <div class="avatar" aria-label="Profile">RR</div>
      </div>
    </header>

    <main class="main">
      <section class="stage" aria-label="Agent stage">
        <div class="stage__background">
          <img class="stage__avatar" src="/robai-photo.png" alt="Agent avatar" />
        </div>

        <div class="stage__visualizer" aria-hidden="true">
          <canvas id="audioViz" class="audio-viz"></canvas>
        </div>

        <div class="controls" role="group" aria-label="Conversation controls">
          <div class="controls__left">
            <div class="mic">
              <button
                id="micButton"
                class="pill-button pill-button--primary"
                aria-label="Mute mic"
                title="Mute mic"
              >
                <span class="icon" aria-hidden="true">${icons.mic}</span>
              </button>
              <span id="micStatus" class="mic__status">Listening</span>
            </div>
            <button
              id="pauseButton"
              class="pill-button pill-button--ghost"
              aria-label="Pause conversation"
              title="Pause conversation"
            >
              <span class="icon" aria-hidden="true">${icons.pause}</span>
            </button>
            <button
              id="leaveButton"
              class="pill-button pill-button--ghost"
              aria-label="Leave session"
              title="Leave session"
            >
              <span class="icon" aria-hidden="true">${icons.leave}</span>
            </button>
            <span class="vertical-divider" aria-hidden="true"></span>
          </div>

          <div class="controls__center">
            <form id="messageForm" class="message-form">
              <input
                id="messageInput"
                type="text"
                placeholder="Write message"
                aria-label="Write message"
              />
              <button class="send-button" type="submit" aria-label="Send" title="Send">
                <span class="icon" aria-hidden="true">${icons.send}</span>
              </button>
            </form>
            <div class="helper-text">AI-generated content may be incorrect</div>
          </div>

          <div class="controls__right">
            <button
              id="transcriptToggle"
              class="pill-button pill-button--ghost"
              type="button"
              aria-label="Open transcript"
              title="Open transcript"
            >
              <span class="icon" aria-hidden="true">${icons.transcript}</span>
            </button>
            <button
              id="soundToggle"
              class="pill-button pill-button--ghost"
              type="button"
              aria-label="Turn off notification sounds"
              title="Turn off notification sounds"
            >
              <span class="icon" aria-hidden="true">${icons.bell}</span>
            </button>
          </div>
        </div>
      </section>

      <aside id="transcriptPanel" class="transcript hidden" aria-label="Transcript">
        <div class="transcript__header">
          <div class="transcript__title">
            <h2>Transcript</h2>
            <div id="transcriptFilter" class="transcript__filter"></div>
          </div>
          <div class="transcript__actions">
            <button
              id="transcriptClose"
              class="pill-button pill-button--ghost"
              aria-label="Close transcript"
              type="button"
              title="Close transcript"
            >
              <span class="icon" aria-hidden="true">${icons.close}</span>
            </button>
          </div>
        </div>
        <div id="transcriptBody" class="transcript__body"></div>
      </aside>
    </main>

    <footer class="footer">
      <span>Privacy policy</span>
      <span>© Microsoft 2026</span>
    </footer>
  </div>
`;
