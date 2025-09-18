package main

import (
	"log"
	"social-flow/tools/cli/src/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		log.Fatalf("❌ CLI execution failed: %v", err)
	}
}
