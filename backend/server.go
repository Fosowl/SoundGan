package main

import (
    "bytes"
    "encoding/json"
    "fmt"
	"log"
    "net/http"
)

func callGanService(w http.ResponseWriter, r *http.Request) (string, error) {
	inputData := map[string]interface{}{
		"output_file": "output.wav",	
	}
    jsonData, err := json.Marshal(inputData)
    if err != nil {
        return "", err
    }
	fmt.Printf("Calling microservice")
    resp, err := http.Post("http://localhost:5050/infer", "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    var result map[string]interface{}
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", err
    }
    image, ok := result["output_path"].(string)
	errorMessage, ok := result["error"].(string)
    if !ok {
        return "", fmt.Errorf("unexpected response format")
    }
	if errorMessage != "" {
		return "", fmt.Errorf(errorMessage)
	}
    return image, nil
}

func generationHandler(w http.ResponseWriter, r *http.Request) {
	image, err := callGanService(w, r)
	fmt.Printf("\nGan generated image: %v \n", image)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write([]byte(image))
}

func main() {
	port := 8080
	fmt.Printf("Starting server at port %v\n", port)
	http.HandleFunc("/generate", generationHandler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}