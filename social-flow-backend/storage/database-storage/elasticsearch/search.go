package elasticsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
)

func SearchVideos(ctx context.Context, query string) ([]map[string]interface{}, error) {
	q := map[string]interface{}{
		"query": map[string]interface{}{
			"multi_match": map[string]interface{}{
				"query":  query,
				"fields": []string{"title", "description", "tags"},
			},
		},
	}

	body, _ := json.Marshal(q)

	res, err := Client.Search(
		Client.Search.WithContext(ctx),
		Client.Search.WithIndex(VideoIndex),
		Client.Search.WithBody(bytes.NewReader(body)),
		Client.Search.WithPretty(),
	)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.IsError() {
		return nil, fmt.Errorf("search error: %s", res.String())
	}

	var r map[string]interface{}
	if err := json.NewDecoder(res.Body).Decode(&r); err != nil {
		return nil, err
	}

	hits := r["hits"].(map[string]interface{})["hits"].([]interface{})
	results := []map[string]interface{}{}
	for _, hit := range hits {
		source := hit.(map[string]interface{})["_source"]
		results = append(results, source.(map[string]interface{}))
	}
	return results, nil
}
