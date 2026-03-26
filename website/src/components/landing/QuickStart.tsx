export default function QuickStart() {
  return (
    <div className="not-content quick-start">
      <h2>Get Running in 3 Steps</h2>
      <div className="quick-start-steps">
        <div className="quick-start-step">
          <div className="step-number">1</div>
          <div className="step-content">
            <h3>Install sparkrun</h3>
            <p>One command installs sparkrun, sets up a managed environment, configures tab completion, and starts the setup wizard.</p>
            <pre className="sl-code"><code>uvx sparkrun setup</code></pre>
          </div>
        </div>
        <div className="quick-start-step">
          <div className="step-number">2</div>
          <div className="step-content">
            <h3>Configure your cluster(s)</h3>
            <p>Setup your DGX Spark or Sparks with our best practices by following the wizard.</p>
            {/*<pre className="sl-code"><code>sparkrun cluster create mylab --hosts 192.168.11.13,192.168.11.14{'\n'}sparkrun cluster set-default mylab</code></pre>*/}
          </div>
        </div>
        <div className="quick-start-step">
          <div className="step-number">3</div>
          <div className="step-content">
            <h3>Run inference</h3>
            <p>Pick a recipe, launch it. Your model is serving — Ctrl+C safely detaches from logs.</p>
            <pre className="sl-code"><code>sparkrun run qwen3-1.7b-vllm</code></pre>
          </div>
        </div>
      </div>
    </div>
  );
}
