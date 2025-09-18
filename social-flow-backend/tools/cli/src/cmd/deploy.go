package cmd

import (
	"fmt"
	"os/exec"

	"github.com/spf13/cobra"
)

var deployCmd = &cobra.Command{
	Use:   "deploy",
	Short: "Deploy infrastructure and services",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("🚀 Deploying services using Docker Compose...")
		out, err := exec.Command("docker-compose", "up", "-d").CombinedOutput()
		if err != nil {
			fmt.Printf("❌ Deployment failed: %v\n", err)
		} else {
			fmt.Println(string(out))
			fmt.Println("✅ Deployment successful!")
		}
	},
}
