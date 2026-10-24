package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// copilotToolSearch is a Copilot account's Messages API as a Business account
// answered on 2026-10-24: tool search asked by Bedrock's and Vertex's beta
// name is turned away whole, and asked by Anthropic's own name it is served.
// The refusal's message and code are those magpie's usage log kept of the
// real one. It keeps each request's anthropic-beta and body.
type copilotToolSearch struct {
	mu     sync.Mutex
	betas  []string
	bodies []map[string]any
}

func (c *copilotToolSearch) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/models" {
		io.WriteString(w, `{"data":[{"id":"claude-opus-5.5","name":"Claude Opus 5.5","model_picker_enabled":true,"policy":{"state":"enabled"},"supported_endpoints":["/v1/messages","/chat/completions"],"capabilities":{"type":"chat"}}]}`)
		return
	}
	b, _ := io.ReadAll(r.Body)
	var body map[string]any
	json.Unmarshal(b, &body)
	beta := r.Header.Get("anthropic-beta")
	c.mu.Lock()
	c.betas = append(c.betas, beta)
	c.bodies = append(c.bodies, body)
	c.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	if slices.Contains(strings.Split(beta, ","), "tool-search-tool-2025-10-19") {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"unsupported beta header(s): tool-search-tool-2025-10-19","code":"invalid_request_body"}}`)
		return
	}
	if strings.HasSuffix(r.URL.Path, "/count_tokens") {
		io.WriteString(w, `{"input_tokens":4242}`)
		return
	}
	io.WriteString(w, `{"id":"m1","type":"message","role":"assistant","model":"claude-opus-5.5","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}`)
}

func (c *copilotToolSearch) asked() ([]string, []map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.betas), slices.Clone(c.bodies)
}

// A turn of Claude Desktop's gateway mode with Enable tool search on, after
// ToolSearch has loaded a deferred MCP tool: the placeholder Claude Code keeps
// deferred, the loaded tool deferred too, and ToolSearch's answer, a
// tool_reference. The shapes and names are those of a real run of the
// Desktop app's Claude Code 2.1.308 on 2026-10-24.
const desktopToolSearchTurn = `{"model":%q,"max_tokens":64,"stream":false,
  "tools":[
    {"name":"ToolSearch","description":"Fetches full schema definitions for deferred tools.","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},
    {"name":"DeferredToolPlaceholder","description":"Reserved placeholder that keeps deferred tool loading active; never call this tool.","input_schema":{"type":"object","properties":{}},"defer_loading":true},
    {"name":"mcp__probe__echo_probe","description":"Echo a text back with a marker.","input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]},"defer_loading":true}],
  "messages":[
    {"role":"user","content":"Call the echo_probe tool of the probe MCP server with the text ping."},
    {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"ToolSearch","input":{"query":"select:mcp__probe__echo_probe","max_results":1}}]},
    {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"tool_reference","tool_name":"mcp__probe__echo_probe"}]}]}]}`

// The betas Claude Desktop's gateway mode asked with Enable tool search on
// (the same real run): tool search by Bedrock's and Vertex's name, as Claude
// Code asks it wherever its experimental betas are off.
const desktopToolSearchBetas = "claude-code-20250219,interleaved-thinking-2025-05-14,tool-search-tool-2025-10-19,effort-2025-11-24"

// Claude Desktop's gateway mode, with Enable tool search on, asks tool
// search as tool-search-tool-2025-10-19, which Copilot turned away on the
// first request of every session (400 "unsupported beta header(s)", a
// Business account, 2026-10-24); Claude Code then turned tool search off
// for the session and sent every tool's schema. A Copilot account is asked
// it by Anthropic's name, in its place, once, with the deferred tools and
// the tool_reference as Claude Code wrote them, and answers, count_tokens
// too; the name Claude Code asks Copilot by at another base URL goes as it
// is, and a provider that isn't Copilot is asked the beta as Claude Code
// named it.
func TestCopilotToolSearchBeta(t *testing.T) {
	up := &copilotToolSearch{}
	copilotFake(t, "gho_toolsearch", up.ServeHTTP)
	relay := &betaRelay{}
	srv := httptest.NewServer(relay)
	t.Cleanup(srv.Close)
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "k", Anthropic: srv.URL,
		Models: []string{"claude-opus-5-5"}}); err != nil {
		t.Fatal(err)
	}
	s := New()
	askAt := func(path, model, betas string) (int, string) {
		t.Helper()
		req := httptest.NewRequest("POST", path, strings.NewReader(fmt.Sprintf(desktopToolSearchTurn, model)))
		req.Header.Set("x-api-key", Token+"-claude-desktop")
		req.Header.Set("User-Agent", "claude-cli/2.1.308 (external, claude-desktop-3p)")
		req.Header.Set("anthropic-beta", betas)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)
		return rec.Code, rec.Body.String()
	}
	ask := func(model, betas string) (int, string) {
		t.Helper()
		return askAt("/v1/messages?beta=true", model, betas)
	}

	code, body := ask("copilot/claude-opus-5.5", desktopToolSearchBetas)
	betas, bodies := up.asked()
	if code != 200 || !strings.Contains(body, "pong") {
		t.Fatalf("Desktop's tool search on Copilot: status %d: %s (Copilot was asked %q)", code, body, betas)
	}
	if want := "claude-code-20250219,interleaved-thinking-2025-05-14,advanced-tool-use-2025-11-20,effort-2025-11-24"; len(betas) != 1 || betas[0] != want {
		t.Fatalf("Copilot was asked %q, want once %q", betas, want)
	}
	sent, _ := json.Marshal(bodies[0])
	for _, keep := range []string{`"defer_loading":true`, `{"tool_name":"mcp__probe__echo_probe","type":"tool_reference"}`} {
		if !strings.Contains(string(sent), keep) {
			t.Errorf("Copilot wasn't sent %s: %s", keep, sent)
		}
	}

	// Claude Code's name for it at another base URL, and the two together
	for _, tc := range []struct{ asked, want string }{
		{"claude-code-20250219,advanced-tool-use-2025-11-20", "claude-code-20250219,advanced-tool-use-2025-11-20"},
		{"tool-search-tool-2025-10-19,advanced-tool-use-2025-11-20", "advanced-tool-use-2025-11-20"},
		{"advanced-tool-use-2025-11-20,tool-search-tool-2025-10-19", "advanced-tool-use-2025-11-20"},
	} {
		if code, body := ask("copilot/claude-opus-5.5", tc.asked); code != 200 {
			t.Fatalf("%q: status %d: %s", tc.asked, code, body)
		}
		if betas, _ := up.asked(); betas[len(betas)-1] != tc.want {
			t.Errorf("%q: Copilot was asked %q, want %q", tc.asked, betas[len(betas)-1], tc.want)
		}
	}

	// Claude Code's count_tokens goes the same way, to Copilot and not to
	// magpie's estimate
	code, body = askAt("/v1/messages/count_tokens?beta=true", "copilot/claude-opus-5.5", desktopToolSearchBetas)
	if code != 200 || !strings.Contains(body, `"input_tokens":4242`) {
		t.Fatalf("count_tokens: status %d: %s", code, body)
	}
	if betas, _ := up.asked(); betas[len(betas)-1] != "claude-code-20250219,interleaved-thinking-2025-05-14,advanced-tool-use-2025-11-20,effort-2025-11-24" {
		t.Errorf("count_tokens: Copilot was asked %q", betas[len(betas)-1])
	}

	// another provider: as Claude Code asked it
	if code, body := ask("relay/claude-opus-5-5", desktopToolSearchBetas); code != 200 {
		t.Fatalf("relay: status %d: %s", code, body)
	}
	if got := relay.last(); !slices.Equal(got, []string{desktopToolSearchBetas}) {
		t.Errorf("relay was asked %q, want %q", got, desktopToolSearchBetas)
	}
}
