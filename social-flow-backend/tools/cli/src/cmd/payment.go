package cmd

import (
	"fmt"
	"social-flow/tools/cli/src/internal"

	"github.com/spf13/cobra"
)

var paymentCmd = &cobra.Command{
	Use:   "payment",
	Short: "Manage payments & monetization",
}

var paymentProcessCmd = &cobra.Command{
	Use:   "process [userID] [amount]",
	Short: "Process a payment",
	Args:  cobra.ExactArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		userID := args[0]
		amount := args[1]
		client := internal.NewAPIClient()
		if err := client.ProcessPayment(userID, amount); err != nil {
			fmt.Printf("❌ Payment failed: %v\n", err)
		} else {
			fmt.Printf("💰 Payment of %s for user %s processed successfully!\n", amount, userID)
		}
	},
}

func init() {
	paymentCmd.AddCommand(paymentProcessCmd)
}
