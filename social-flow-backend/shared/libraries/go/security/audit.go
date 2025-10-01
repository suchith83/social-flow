package security

import (
	"log"
	"os"
	"time"
)

var auditLogger *log.Logger

func init() {
	file, err := os.OpenFile("audit.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		panic("failed to open audit log: " + err.Error())
	}
	auditLogger = log.New(file, "AUDIT: ", log.LstdFlags|log.Lmicroseconds)
}

// LogAudit logs security-critical events.
func LogAudit(event, userID string) {
	auditLogger.Printf("[%s] user=%s event=%s", time.Now().Format(time.RFC3339), userID, event)
}
