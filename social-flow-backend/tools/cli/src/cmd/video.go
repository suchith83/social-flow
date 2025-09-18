package cmd

import (
	"fmt"
	"social-flow/tools/cli/src/internal"

	"github.com/spf13/cobra"
)

var videoCmd = &cobra.Command{
	Use:   "video",
	Short: "Manage videos",
}

var videoUploadCmd = &cobra.Command{
	Use:   "upload [file]",
	Short: "Upload a new video",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		filePath := args[0]
		client := internal.NewAPIClient()
		if err := client.UploadVideo(filePath); err != nil {
			fmt.Printf("❌ Upload failed: %v\n", err)
		} else {
			fmt.Printf("🎥 Video uploaded successfully: %s\n", filePath)
		}
	},
}

func init() {
	videoCmd.AddCommand(videoUploadCmd)
}
