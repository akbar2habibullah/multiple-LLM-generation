import kMeansText from 'kmeans-categorize-text'
import cosineSimilarity from 'compute-cosine-similarity'
import { GoogleGenerativeAI } from "@google/generative-ai"
import Groq from "groq-sdk"

const groq = new Groq({ apiKey: process.env.GOOGLE_GEMINI_API_KEY })
const genAI = new GoogleGenerativeAI(process.env.GROQ_API_KEY)
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" })



const generateResponses = async (prompt, numResponses = 5, model = 'llama3-8b-8192') => {
  const responses = []
  for (let i = 0; i < numResponses; i++) {
    const chatCompletion = await groq.chat.completions.create({
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
      model,
      max_tokens: 100,
      temperature: 0.5
    })

    const result = chatCompletion.choices[0]?.message?.content
    responses.push(result)
  }
  return responses
}

const generateResponse = async (prompt, model = 'llama3-70b-8192') => {
  const chatCompletion = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
    model,
    max_tokens: 100,
    temperature: 0.5
  })
  const result = chatCompletion.choices[0]?.message?.content

  return result || ""
}

const getEmbeddings = async (texts) => {
  let results = []

  for (let i = 0; i < texts.length; i++) {
    const result = await embeddingModel.embedContent(texts[i])
    const tmp = result.embedding.values

    results.push(tmp)
  }

  return results
}

const findUniqueResponses = (responses, numClusters = 3, func) => {
  kMeansText(responses.map((response, idx) => {
    return { id: idx, text: response }
  }), numClusters, [], result => {
    func(result)
  }, error => console.log(error))
}

const generateComprehensiveJudgment = async (uniqueResponses, input) => {
  let combinedPrompt = "Please consider these sets of ideas and make one last judgment:\n"

  let num = 1
  for (let i in uniqueResponses) {
    combinedPrompt += `Response ${num}: ${uniqueResponses[i][Object.keys(uniqueResponses[i])[0]]}\n`
    num += 1
  }

  combinedPrompt += `Context: ${input}\n`

  console.log('----------------')
  console.log(combinedPrompt)
  console.log('----------------')

  const response = await generateResponse(combinedPrompt)

  return response
}

const calculateAttentionScores = (embeddings) => {
  const similarityMatrix = embeddings.map((embed1) =>
    embeddings.map((embed2) => cosineSimilarity(embed1, embed2))
  )

  const attentionScores = similarityMatrix.map((row) => {
    const rowSum = row.reduce((sum, value) => sum + value, 0)
    return row.map(value => value / rowSum)
  })

  return attentionScores
}

const findMostImportantResponse = (responses, embeddings, attentionScores) => {
  const weightedEmbeddings = embeddings.map((_, i) => {
    return embeddings.reduce((sum, embed, j) => {
      return sum.map((val, k) => val + attentionScores[i][j] * embed[k])
    }, new Array(embeddings[0].length).fill(0))
  })

  const distances = embeddings.map((embed, i) => {
    return Math.sqrt(embed.reduce((sum, val, k) => {
      return sum + Math.pow(val - weightedEmbeddings[i][k], 2)
    }, 0))
  })

  const mostImportantResponseIdx = distances.indexOf(Math.min(...distances))
  return responses[mostImportantResponseIdx]
}

// Example usage
const prompt = "Do you know Hu Tao?"
generateResponses(prompt, 12).then(responses => {
  getEmbeddings(responses).then(embeddings => {
    findUniqueResponses(responses, 5, (uniqueResponses) => {
      generateComprehensiveJudgment(uniqueResponses, prompt).then(judgment => {
        console.log("Comprehensive judgment based on unique responses:")
        console.log(judgment)
      })
    })

    const attentionScores = calculateAttentionScores(embeddings)

    const mostImportantResponse = findMostImportantResponse(responses, embeddings, attentionScores)
    console.log("\nMost important response based on self-attention mechanism:")
    console.log(mostImportantResponse)
  })
})
