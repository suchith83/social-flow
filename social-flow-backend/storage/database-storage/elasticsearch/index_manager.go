package elasticsearch

import (
	"bytes"
	"context"
	"fmt"
	"log"
)

// EnsureIndex creates index with mappings if not exists
func EnsureIndex(ctx context.Context, index string, mapping string) error {
	exists, err := Client.Indices.Exists([]string{index})
	if err != nil {
		return err
	}
	if exists.StatusCode == 200 {
		log.Printf("ℹ️ Index [%s] already exists.", index)
		return nil
	}

	res, err := Client.Indices.Create(
		index,
		Client.Indices.Create.WithBody(bytes.NewReader([]byte(mapping))),
		Client.Indices.Create.WithContext(ctx),
	)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("failed to create index %s: %s", index, res.String())
	}
	log.Printf("✅ Index [%s] created successfully.", index)
	return nil
}
