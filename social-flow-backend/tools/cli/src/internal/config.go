package internal

import (
	"fmt"
	"os"

	"github.com/spf13/viper"
)

func LoadConfig(cfgFile string) error {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, _ := os.UserHomeDir()
		viper.AddConfigPath(home)
		viper.SetConfigName(".sfcli")
		viper.SetConfigType("yaml")
	}

	viper.AutomaticEnv() // Read from environment as well

	if err := viper.ReadInConfig(); err != nil {
		return fmt.Errorf("config error: %w", err)
	}
	return nil
}
