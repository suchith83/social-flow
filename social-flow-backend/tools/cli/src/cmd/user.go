package cmd

import (
	"fmt"
	"social-flow/tools/cli/src/internal"

	"github.com/spf13/cobra"
)

var userCmd = &cobra.Command{
	Use:   "user",
	Short: "Manage users",
}

var userCreateCmd = &cobra.Command{
	Use:   "create [username]",
	Short: "Create a new user",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		username := args[0]
		client := internal.NewAPIClient()
		if err := client.CreateUser(username); err != nil {
			fmt.Printf("❌ Failed to create user: %v\n", err)
		} else {
			fmt.Printf("✅ User %s created successfully!\n", username)
		}
	},
}

var userGetCmd = &cobra.Command{
	Use:   "get [userID]",
	Short: "Fetch user details",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		userID := args[0]
		client := internal.NewAPIClient()
		user, err := client.GetUser(userID)
		if err != nil {
			fmt.Printf("❌ Failed to fetch user: %v\n", err)
		} else {
			fmt.Printf("👤 User: %+v\n", user)
		}
	},
}

func init() {
	userCmd.AddCommand(userCreateCmd)
	userCmd.AddCommand(userGetCmd)
}
