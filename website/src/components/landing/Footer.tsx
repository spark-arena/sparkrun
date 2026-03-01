export default function Footer() {
  return (
    <footer className="not-content landing-footer">
      <div className="footer-grid">
        <div className="footer-brand">
          <h3>sparkrun</h3>
          <p>One command to rule them all. Launch, manage, and stop inference workloads on NVIDIA DGX Spark systems.</p>
        </div>
        <div className="footer-column">
          <h4>Docs</h4>
          <ul>
            <li><a href="/getting-started/installation/">Installation</a></li>
            <li><a href="/getting-started/quick-start/">Quick Start</a></li>
            <li><a href="/cli/overview/">CLI Reference</a></li>
            <li><a href="/recipes/format/">Recipe Format</a></li>
          </ul>
        </div>
        <div className="footer-column">
          <h4>Runtimes</h4>
          <ul>
            <li><a href="/runtimes/vllm/">vLLM</a></li>
            <li><a href="/runtimes/sglang/">SGLang</a></li>
            <li><a href="/runtimes/llama-cpp/">llama.cpp</a></li>
            <li><a href="/runtimes/eugr-vllm/">eugr-vllm</a></li>
          </ul>
        </div>
        <div className="footer-column">
          <h4>Community</h4>
          <ul>
            <li><a href="https://github.com/scitrera/sparkrun">GitHub</a></li>
            <li><a href="https://github.com/scitrera/sparkrun/issues">Issues</a></li>
            <li><a href="https://pypi.org/project/sparkrun/">PyPI</a></li>
          </ul>
        </div>
      </div>
      <div className="footer-bottom">
        <p>Apache 2.0 License</p>
        <div className="footer-powered">
          Part of the <a href="https://spark-arena.com">Spark Arena</a> ecosystem &middot; Powered by <a href="https://scitrera.ai">scitrera.ai</a>
        </div>
      </div>
    </footer>
  );
}
