package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"social-flow/tools/cli/src/internal"
)

var cfgFile string

// rootCmd is the base command.
var rootCmd = &cobra.Command{
	Use:   "sfcli",
	Short: "Social Flow CLI - Manage services, users, videos, and infra",
	Long:  `Enterprise-grade CLI for Social Flow platform. Provides commands for users, videos, payments, and deployments.`,
}

// Execute starts the CLI.
func Execute() error {
	return rootCmd.Execute()
}

func init() {
	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.sfcli.yaml)")

	// Register subcommands
	rootCmd.AddCommand(userCmd)
	rootCmd.AddCommand(videoCmd)
	rootCmd.AddCommand(paymentCmd)
	rootCmd.AddCommand(deployCmd)
}

func initConfig() {
	if err := internal.LoadConfig(cfgFile); err != nil {
		fmt.Printf("⚠️  Failed to load config: %v\n", err)
		os.Exit(1)
	}
}
