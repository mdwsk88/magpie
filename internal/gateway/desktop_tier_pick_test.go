package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// desktopPickSetup is a provider serving m1 and claude-sonnet-5-5, both
// with reasoning levels (so Desktop lists them as mythos-magpie-<n> and
// magpie-<n>.anthropic.claude-sonnet-5-5), on a fake Anthropic upstream
// that notes the model of the last request.
func desktopPickSetup(t *testing.T) *desktopSubagentUpstream {
	t.Helper()
	fresh(t)
	noRests()
	t.Cleanup(noRests)
	desktopPicked.Lock()
	desktopPicked.model, desktopPicked.from = "", ""
	desktopPicked.Unlock()
	u := &desktopSubagentUpstream{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		model := modelOf(body)
		u.mu.Lock()
		u.model, u.path = model, r.URL.Path
		u.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"reply","type":"message","role":"assistant","model":"`+model+`","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":1}}`)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Anthropic: up.URL, Models: []string{"m1", "claude-sonnet-5-5"}}); err != nil {
		t.Fatal(err)
	}
	for _, ref := range []string{"fake/m1", "fake/claude-sonnet-5-5"} {
		if err := provider.SetModelEfforts(ref, []string{"low", "medium", "high"}); err != nil {
			t.Fatal(err)
		}
	}
	return u
}

func shownEntry(t *testing.T, match func(provider.Entry) bool) provider.Entry {
	t.Helper()
	shown, _ := provider.CatalogFor("claude-desktop")
	for _, e := range shown {
		if match(e) {
			return e
		}
	}
	t.Fatalf("not shown to Claude Desktop: %v", shown)
	return provider.Entry{}
}

// tierEntries are the entries of an inferenceModels list by the tier they
// are tagged with; a tier tagged twice fails.
func tierEntries(t *testing.T, list []map[string]any) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	for _, e := range list {
		if tier, _ := e["anthropicFamilyTier"].(string); tier != "" {
			if out[tier] != nil {
				t.Fatalf("%s tagged twice: %v", tier, list)
			}
			out[tier] = e
		}
	}
	return out
}

// A model picked for two of Claude Desktop's tiers (a user's
// copilot/claude-sonnet-5.5 for sonnet and haiku) was tagged with the
// first only: Desktop keys inferenceModels by name and reads one tier off
// an entry, so haiku had no model, Claude Code in the Code tab got an empty
// ANTHROPIC_DEFAULT_HAIKU_MODEL, and its small tasks (WebFetch's page
// summaries) ran on the session's model. The second tier is listed once
// more, after the rows, by the model's id form under a number of its own,
// which the gateway reads back to the model and serves as the pick.
func TestDesktopTierSharedPick(t *testing.T) {
	t.Run("claude model", func(t *testing.T) {
		u := desktopPickSetup(t)
		sonnet := shownEntry(t, func(e provider.Entry) bool { return e.ID == "fake/claude-sonnet-5-5" })
		for _, tier := range []string{"sonnet", "haiku"} {
			if err := SetDesktopTier(tier, sonnet.ID); err != nil {
				t.Fatal(err)
			}
		}
		list, ok := DesktopInferenceModels()
		shown, _ := provider.CatalogFor("claude-desktop")
		rows := desktopModels(shown, true)
		if !ok || len(list) != len(rows)+1 {
			t.Fatalf("%d entries for %d rows and one more tier: %v", len(list), len(rows), list)
		}
		for i, r := range rows {
			if want := strings.TrimSuffix(r["id"].(string), "[1m]"); list[i]["name"] != want {
				t.Fatalf("entry %d is %v, /v1/models has %s there", i, list[i]["name"], want)
			}
		}
		tiers := tierEntries(t, list)
		if e := tiers["sonnet"]; e == nil || e["name"] != DesktopID(sonnet) {
			t.Fatalf("sonnet: %v", list)
		}
		haiku := tiers["haiku"]
		if haiku == nil {
			t.Fatalf("haiku has no entry, so Claude Code's ANTHROPIC_DEFAULT_HAIKU_MODEL is empty: %v", list)
		}
		alias, _ := haiku["name"].(string)
		if !regexp.MustCompile(`^magpie-\d{10}\.anthropic\.claude-sonnet-5-5$`).MatchString(alias) || alias == DesktopID(sonnet) ||
			haiku["isFamilyDefault"] != true || list[len(list)-1]["name"] != alias {
			t.Fatalf("haiku's entry: %v (sonnet's name %s)", haiku, DesktopID(sonnet))
		}
		if _, has := haiku["behavesAs"]; has {
			t.Fatalf("Claude Code knows %s as Claude Sonnet 5.5: %v", alias, haiku)
		}
		if l, _ := haiku["labelOverride"].(string); !strings.HasSuffix(l, " · haiku") {
			t.Fatalf("haiku's label: %v", haiku)
		}
		for _, id := range []string{alias, alias + "[1m]"} {
			if got := DesktopCatalogID(id); got != sonnet.ID {
				t.Fatalf("%s read back as %q", id, got)
			}
		}
		if a := DesktopTierAliases(); len(a) != 1 || len(a[sonnet.ID]) != 1 || a[sonnet.ID][0] != alias {
			t.Fatalf("tier aliases: %v", a)
		}

		// the session on m1; Claude Code's small task for the haiku tier,
		// by that entry's name, goes to the pick, not the session's model
		s := New()
		m1 := shownEntry(t, func(e provider.Entry) bool { return e.ID == "fake/m1" })
		desktopSubagentAsk(t, s, Token+"-claude-desktop", DesktopID(m1), "", "/v1/messages", true)
		if m, _ := u.seen(); m != "m1" {
			t.Fatalf("the session's turn went to %q", m)
		}
		desktopSubagentAsk(t, s, Token+"-claude-desktop", alias, "", "/v1/messages", false)
		if m, _ := u.seen(); m != "claude-sonnet-5-5" {
			t.Fatalf("haiku's small task went to %q, want the haiku pick claude-sonnet-5-5", m)
		}

		// the picks apart again: one entry a tier, none more
		if err := SetDesktopTier("haiku", "fake/m1"); err != nil {
			t.Fatal(err)
		}
		if list, _ := DesktopInferenceModels(); len(list) != len(rows) {
			t.Fatalf("picks apart: %v", list)
		}
		if a := DesktopTierAliases(); a != nil {
			t.Fatalf("picks apart, tier aliases: %v", a)
		}
	})

	t.Run("other model", func(t *testing.T) {
		desktopPickSetup(t)
		m1 := shownEntry(t, func(e provider.Entry) bool { return e.ID == "fake/m1" })
		for _, tier := range []string{"opus", "haiku"} {
			if err := SetDesktopTier(tier, m1.ID); err != nil {
				t.Fatal(err)
			}
		}
		list, _ := DesktopInferenceModels()
		tiers := tierEntries(t, list)
		opus, haiku := tiers["opus"], tiers["haiku"]
		if opus == nil || opus["name"] != DesktopID(m1) || opus["behavesAs"] != "claude-opus-5-5" {
			t.Fatalf("opus: %v", list)
		}
		if haiku == nil {
			t.Fatalf("haiku has no entry: %v", list)
		}
		alias, _ := haiku["name"].(string)
		if !regexp.MustCompile(`^mythos-magpie-\d{10}$`).MatchString(alias) || alias == DesktopID(m1) || haiku["behavesAs"] != "claude-haiku-5-5" {
			t.Fatalf("haiku's entry: %v", haiku)
		}
		as, ok := DesktopBehavesAs()
		if !ok || as[DesktopID(m1)] != "claude-opus-5-5" || as[alias] != "claude-haiku-5-5" || len(as) != 2 {
			t.Fatalf("behaves as: %v %v", as, ok)
		}
		if got := DesktopCatalogID(alias); got != m1.ID {
			t.Fatalf("%s read back as %q", alias, got)
		}
	})
}

// A routing group led by a Claude model, picked for a tier, is shown to
// Desktop by its mythos-magpie-<n> (claudeLooking: never as the model
// itself), which Claude Code doesn't know; its entry had no behavesAs, as
// the group's model is a Claude model, so the tier ran on Claude Code's
// generic prompt and defaults. It runs as the group's model.
func TestDesktopGroupPickRunsAsItsModel(t *testing.T) {
	desktopPickSetup(t)
	if err := provider.SaveGroup(provider.Group{ID: "duo", Name: "Duo", Members: []string{"fake/claude-sonnet-5-5"}}); err != nil {
		t.Fatal(err)
	}
	g := shownEntry(t, func(e provider.Entry) bool { return e.Group == "duo" })
	if err := SetDesktopTier("haiku", g.ID); err != nil {
		t.Fatal(err)
	}
	list, _ := DesktopInferenceModels()
	haiku := tierEntries(t, list)["haiku"]
	if haiku == nil || haiku["name"] != DesktopID(g) {
		t.Fatalf("the group for haiku: %v", list)
	}
	if strings.HasPrefix(DesktopID(g), "magpie-") || haiku["behavesAs"] != "claude-sonnet-5-5" {
		t.Fatalf("the group, shown as %s, runs as %v; want claude-sonnet-5-5: %v", DesktopID(g), haiku["behavesAs"], haiku)
	}
	if as, _ := DesktopBehavesAs(); as[DesktopID(g)] != "claude-sonnet-5-5" {
		t.Fatalf("behaves as: %v", as)
	}
}
