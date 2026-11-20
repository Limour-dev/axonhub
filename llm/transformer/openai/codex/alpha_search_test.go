package codex

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func TestAlphaSearchRequestSessionIDPrecedence(t *testing.T) {
	tests := []struct {
		name            string
		headers         http.Header
		body            string
		contextSession  string
		expectedSession string
	}{
		{
			name:            "header before body and context",
			headers:         http.Header{SessionHeaderHyphen: []string{"header-session"}},
			body:            `{"id":"body-session","model":"client-model","commands":{"search_query":[]}}`,
			contextSession:  "context-session",
			expectedSession: "header-session",
		},
		{
			name:            "body before context",
			headers:         http.Header{},
			body:            `{"id":"body-session","model":"client-model","commands":{"search_query":[]}}`,
			contextSession:  "context-session",
			expectedSession: "body-session",
		},
		{
			name:            "context fallback",
			headers:         http.Header{},
			body:            `{"model":"client-model","commands":{"search_query":[]}}`,
			contextSession:  "context-session",
			expectedSession: "context-session",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := shared.WithSessionID(context.Background(), tt.contextSession)
			outbound := newAlphaSearchTestTransformer(t)

			request, err := outbound.TransformRequest(ctx, &llm.Request{
				Model:       "mapped-model",
				RequestType: llm.RequestTypeAlphaSearch,
				APIFormat:   llm.APIFormatOpenAIAlphaSearch,
				RawRequest:  &httpclient.Request{Headers: tt.headers},
				AlphaSearch: &llm.AlphaSearchRequest{Body: []byte(tt.body)},
			})
			require.NoError(t, err)

			require.Equal(t, "https://chatgpt.com/backend-api/codex/alpha/search", request.URL)
			require.Equal(t, tt.expectedSession, request.Headers.Get(SessionHeaderHyphen))
			require.Equal(t, tt.expectedSession, request.Headers.Get(ThreadIDHeader))
			require.Equal(t, tt.expectedSession+":0", request.Headers.Get(WindowIDHeader))
			require.NotEmpty(t, request.Headers.Get(ClientRequestIDHeader))
			require.Equal(t, fabricatedBetaFeatures, request.Headers.Get(BetaFeaturesHeader))
			require.Equal(t, AxonHubOriginator, request.Headers.Get("Originator"))
			require.Equal(t, codexDefaultVersion, request.Headers.Get("Version"))
			require.Equal(t, testChatAccountID, request.Headers.Get("Chatgpt-Account-Id"))
			require.Equal(t, "application/json", request.Headers.Get("Content-Type"))
			require.Equal(t, "application/json", request.Headers.Get("Accept"))
			require.Equal(t, httpclient.AuthTypeBearer, request.Auth.Type)

			var metadata TurnMetadata
			require.NoError(t, json.Unmarshal([]byte(request.Headers.Get(TurnMetadataHeader)), &metadata))
			require.Equal(t, tt.expectedSession, metadata.SessionID)
			require.Equal(t, tt.expectedSession, metadata.ThreadID)

			var body map[string]any
			require.NoError(t, json.Unmarshal(request.Body, &body))
			require.Equal(t, "mapped-model", body["model"])
		})
	}
}

func TestAlphaSearchRequestTurnIDFromAHMetadata(t *testing.T) {
	outbound := newAlphaSearchTestTransformer(t)
	inbound := &llm.Request{
		Model:       "mapped-model",
		RequestType: llm.RequestTypeAlphaSearch,
		APIFormat:   llm.APIFormatOpenAIAlphaSearch,
		RawRequest: &httpclient.Request{Headers: http.Header{
			AHMetadataTurnIDHeader: {"ah-turn-123"},
		}},
		AlphaSearch: &llm.AlphaSearchRequest{Body: []byte(`{"id":"body-session","model":"client-model","commands":{"search_query":[]}}`)},
	}

	// Retries transform the same request again, so merging must not consume the ID.
	for range 2 {
		request, err := outbound.TransformRequest(context.Background(), inbound)
		require.NoError(t, err)
		merged := httpclient.MergeInboundRequest(request, inbound.RawRequest)

		var metadata TurnMetadata
		require.NoError(t, json.Unmarshal([]byte(merged.Headers.Get(TurnMetadataHeader)), &metadata))
		require.Equal(t, "ah-turn-123", metadata.TurnID)
		require.Equal(t, "body-session", metadata.SessionID)
		require.Empty(t, merged.Headers.Values(AHMetadataTurnIDHeader))
	}
}

// TestAlphaSearchDoesNotForwardInboundUserAgent mirrors the Response path:
// the alpha-search request presents the Codex default UA and never copies the
// inbound client UA (issue #2635).
func TestAlphaSearchDoesNotForwardInboundUserAgent(t *testing.T) {
	ctx := context.Background()
	outbound := newAlphaSearchTestTransformer(t)

	const clientUA = "ZCode/3.14.4 (darwin arm64)"

	request, err := outbound.TransformRequest(ctx, &llm.Request{
		Model:       "mapped-model",
		RequestType: llm.RequestTypeAlphaSearch,
		APIFormat:   llm.APIFormatOpenAIAlphaSearch,
		RawRequest: &httpclient.Request{Headers: http.Header{
			"User-Agent": []string{clientUA},
		}},
		AlphaSearch: &llm.AlphaSearchRequest{Body: []byte(`{"model":"client-model","commands":{"search_query":[]}}`)},
	})
	require.NoError(t, err)

	require.Equal(t, codexDefaultUserAgent, request.Headers.Get("User-Agent"))
	require.NotEqual(t, clientUA, request.Headers.Get("User-Agent"))
}

func newAlphaSearchTestTransformer(t *testing.T) *OutboundTransformer {
	t.Helper()

	outbound, err := NewOutboundTransformer(Params{
		BaseURL: "https://chatgpt.com/backend-api/codex#",
		TokenProvider: staticTokenGetter{creds: &oauth.OAuthCredentials{
			AccessToken: testAccessTokenWithAccountID(t),
			ExpiresAt:   time.Now().Add(time.Hour),
		}},
	})
	require.NoError(t, err)

	return outbound
}
