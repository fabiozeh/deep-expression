curl -X POST https://api.notion.com/v1/pages \
  -H "Authorization: Bearer secret_70fQSjf7GVb1oG3zNfiYrZaqkgxjm8udQcmyR74KVnE" \
  -H "Content-Type: application/json" \
  -H "Notion-Version: 2021-05-13" \
  --data '{
    "parent": { "database_id": "331c5ac70a524f8b9604f775d462b3d4" },
    "properties": {
      "Job": {
        "id":"\\S:r",
        "type": "rich_text",
        "rich_text": [{
          "type": "text",
          "text": {
            "content": "'$2'"
          }
        }]
      },
      "Val Rel MSE": {
        "id":"ojUW",
        "type":"rich_text",
        "rich_text": [{
          "type": "text",
          "text": {
            "content": "'$3'"
          }
        }]
      },
      "encoding/feats": {
        "id":"title",
        "type":"title",
        "title": [
        {
          "type": "text",
          "text": {
            "content": "'$1'"
          },
          "annotations": {
            "bold": false,
            "italic": false,
            "strikethrough": false,
            "underline": false,
            "code": false,
            "color": "default"
          }
        }
      ]
      }
    }
  }'
