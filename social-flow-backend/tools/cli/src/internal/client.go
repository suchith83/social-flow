package internal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type APIClient struct {
	BaseURL string
	Client  *http.Client
}

func NewAPIClient() *APIClient {
	return &APIClient{
		BaseURL: "http://localhost:8080/api/v1",
		Client:  &http.Client{},
	}
}

func (c *APIClient) CreateUser(username string) error {
	body, _ := json.Marshal(map[string]string{"username": username})
	resp, err := c.Client.Post(c.BaseURL+"/users", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status: %s", resp.Status)
	}
	return nil
}

func (c *APIClient) GetUser(userID string) (map[string]interface{}, error) {
	resp, err := c.Client.Get(c.BaseURL + "/users/" + userID)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var user map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&user); err != nil {
		return nil, err
	}
	return user, nil
}

func (c *APIClient) UploadVideo(filePath string) error {
	// TODO: Implement multipart upload (simulate for now)
	fmt.Printf("📤 Simulating upload of %s to server...\n", filePath)
	return nil
}

func (c *APIClient) ProcessPayment(userID string, amount string) error {
	body, _ := json.Marshal(map[string]string{"user_id": userID, "amount": amount})
	resp, err := c.Client.Post(c.BaseURL+"/payments", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %s", resp.Status)
	}
	return nil
}
