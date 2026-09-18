"""Mocked transport and synthetic chapter metadata; no lessons or learner."""
from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from experiments import definition_tutor_author as author
WIRE_REQUEST = author._request_json


class VerifiedTutorAuthorTests(unittest.TestCase):
    work = dict(author_invocations=0, mock_api_calls=0, mock_metadata_calls=0,
                mock_chat_calls=0, mock_returned_chats=0, real_network_calls=0,
                generated_lessons=0, learner_models=0, optimizer_updates=0, fake_http_connections=0)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        blocker = mock.patch("http.client.HTTPConnection", side_effect=AssertionError("real network forbidden"))
        blocker.start(); self.addCleanup(blocker.stop)
        transport = mock.patch.object(author, "_request_json", side_effect=AssertionError("unmocked transport forbidden"))
        self.transport = transport.start(); self.addCleanup(transport.stop)

    def request(self, mode="local"):
        contract=author.compiler.make_contract("a"*64)
        metrics={key:dict(count=32 if key=="all_query_pair_both" else 320 if key.startswith("known") else 64,
                          total=32 if key=="all_query_pair_both" else 320 if key.startswith("known") else 64) for key in author.METRICS}
        current=dict(weights_sha256="1"*64,lifetime_updates=1944,
            panels={p:{f:deepcopy(metrics) for f in author.FAMILIES} for p in author.PANELS})
        return dict(schema=author.REQUEST_SCHEMA,
            parent=dict(identity_sha256="2"*64,weights_sha256="1"*64,cycle=1,lifetime_updates=1944),
            contract=contract,procedural_recipe=author.compiler.procedural_recipe(contract),
            development=dict(schema=author.EVIDENCE_SCHEMA,role="development",current=current,reference=deepcopy(current)),
            mode=mode,teacher=dict(model="fixture:tiny",sha256="3"*64) if mode=="local" else None,
            max_seconds=10,source_sha256=author.source_hashes())

    def recipe(self, request):
        recipe=deepcopy(request["procedural_recipe"])
        field=next(key for key in recipe if key!="schema")
        recipe[field]=["bind_c_s","revise_c_s","bind_s_d","revise_s_d","bind_c_d","revise_c_d"]
        return recipe

    def service(self, request, *, content=None, error=None, before="3"*64, after="3"*64, done=True):
        calls, tags = [], 0
        if content is None: content = json.dumps(self.recipe(request), separators=(",",":"))
        def call(route,payload,deadline):
            nonlocal tags
            self.work["mock_api_calls"] += 1
            calls.append((route,deepcopy(payload),deadline))
            if route=="/api/tags":
                tags += 1; self.work["mock_metadata_calls"] += 1
                return dict(models=[dict(name="fixture:tiny",digest=before if tags==1 else after)])
            if route=="/api/show":
                self.work["mock_metadata_calls"] += 1
                return dict(details={"family":"fixture"})
            self.assertEqual(route,"/api/chat")
            self.work["mock_chat_calls"] += 1
            self.assertEqual(payload["keep_alive"],0)
            self.assertIs(payload["stream"],False)
            self.assertEqual(payload["options"]["num_ctx"],4096)
            self.assertEqual(set(payload["format"]["properties"]),set(request["procedural_recipe"]))
            self.assertEqual({c[2] for c in calls},{deadline})
            self.assertTrue((self.root/self.slot/"intent.json").exists())
            if error is not None: raise error
            self.work["mock_returned_chats"] += 1
            return dict(model="fixture:tiny",done=done,done_reason="stop",
                message=dict(role="assistant",content=content),prompt_eval_count=300,eval_count=60,
                total_duration=100,load_duration=10,prompt_eval_duration=20,eval_duration=60)
        self.transport.side_effect = call
        return calls

    def invoke(self, request, slot="slot", **kwargs):
        self.slot=slot;self.work["author_invocations"] += 1
        return author.author_curriculum(self.root/slot,request=request,
            expected_request_sha256=author.request_sha256(request),**kwargs)

    def test_accepted_recipe_is_durable_reusable_and_tamper_evident(self):
        request=self.request();original=deepcopy(request)
        calls=self.service(request);result=self.invoke(request)
        saved=result["result"]
        self.assertEqual(saved["outcome"],"local_accepted")
        self.assertEqual(saved["recipe"],self.recipe(request))
        self.assertEqual(saved["recipe_sha256"],author.request_sha256(saved["recipe"]))
        raw=(self.root/"slot"/"response.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),saved["teacher_cost"]["response_file_sha256"])
        self.assertEqual(json.loads(raw)["message"]["content"],json.dumps(self.recipe(request),separators=(",",":")))
        self.assertEqual(request,original)
        before=len(calls)
        with self.assertRaises(author.DecisionPinRequired):self.invoke(request)
        reused=self.invoke(request,expected_result_sha256=result["result_sha256"])
        self.assertTrue(reused["reused"]);self.assertEqual(reused["result"],saved)
        self.assertEqual(reused["invocation_cost"]["api_attempts"],0);self.assertEqual(len(calls),before)
        (self.root/"slot"/"response.json").write_bytes(raw+b" ")
        with self.assertRaises(ValueError):
            author.load_result(self.root/"slot"/"result.json",expected_sha256=result["result_sha256"],
                               expected_request_sha256=author.request_sha256(request))

    def test_invalid_and_injected_orders_fall_back_with_original_response(self):
        request=self.request();field=next(k for k in request["procedural_recipe"] if k!="schema")
        injected=self.recipe(request);injected["answer"]="Yes."
        unknown=self.recipe(request);unknown[field][0]="new_skill"
        duplicate=self.recipe(request);duplicate[field][1]=duplicate[field][0]
        reversed_pair=self.recipe(request);reversed_pair[field][:2]=reversed_pair[field][:2][::-1]
        for i,content in enumerate((json.dumps(injected),json.dumps(unknown),json.dumps(duplicate),json.dumps(reversed_pair),'{"schema":"x","schema":"y"}')):
            with self.subTest(i=i):
                self.service(request,content=content);returned=self.invoke(request,str(i));result=returned["result"]
                self.assertEqual(result["outcome"],"procedural_fallback")
                self.assertEqual(result["recipe"],request["procedural_recipe"])
                self.assertEqual(result["teacher_cost"]["chat_completions"],1)
                self.assertEqual(result["rejection_reason"],"recipe_unavailable_or_invalid")
                self.assertEqual(author.load_result(self.root/str(i)/"result.json",
                    expected_sha256=returned["result_sha256"],expected_request_sha256=author.request_sha256(request)),result)

    def test_unknown_timeout_interrupt_and_incomplete_response_never_repeat(self):
        request=self.request()
        for i,(error,done,exception) in enumerate(((TimeoutError("fixture"),True,author.UnknownTutorRequest),
                (KeyboardInterrupt("fixture"),True,KeyboardInterrupt),(None,False,author.UnknownTutorRequest))):
            with self.subTest(i=i):
                calls=self.service(request,error=error,done=done)
                with self.assertRaises(exception) as raised:self.invoke(request,str(i))
                self.assertTrue(raised.exception.tutor_report["teacher_cost"]["physical_teacher_work_unknown"])
                self.assertFalse((self.root/str(i)/"result.json").exists())
                self.assertTrue((self.root/str(i)/"uncommitted.json").exists())
                before=len(calls)
                with self.assertRaises(author.UnknownTutorRequest):self.invoke(request,str(i))
                self.assertEqual(len(calls),before)

    def test_procedural_and_withdrawal_make_zero_calls_even_on_reuse(self):
        for mode in ("procedural","withdrawn"):
            request=self.request(mode);returned=self.invoke(request,mode)
            result=returned["result"]
            self.assertEqual(result["outcome"],"teacher_withdrawn" if mode=="withdrawn" else "procedural")
            self.assertEqual(result["recipe"],request["procedural_recipe"])
            self.assertEqual(result["teacher_cost"]["api_attempts"],0)
            self.assertTrue(self.invoke(request,mode,expected_result_sha256=returned["result_sha256"])["reused"])
        self.transport.assert_not_called()

    def test_parent_development_source_contract_and_counts_rejected_before_intent(self):
        original=self.request()
        cases=("audit","parent","source","contract","recipe","hash","known_total","impossible_solved","false_zero_solved","reference_denominator","raw_examples")
        for case in cases:
            request=deepcopy(original);metrics=request["development"]["current"]["panels"]["binding"]["color"]
            if case=="audit":request["development"]["role"]="audit"
            elif case=="parent":request["parent"]["weights_sha256"]="9"*64
            elif case=="source":request["source_sha256"]["experiments/definition_tutor_author.py"]="0"*64
            elif case=="contract":request["contract"]["updates"]=107
            elif case=="recipe":request["procedural_recipe"]=self.recipe(request)
            elif case=="known_total":metrics["known_action"].update(count=64,total=64);metrics["known_reply"].update(count=64,total=64)
            elif case=="impossible_solved":metrics["known_action"]["count"]=319
            elif case=="false_zero_solved":metrics["all_query_pair_both"]["count"]=0
            elif case=="reference_denominator":
                for panels in request["development"]["reference"]["panels"].values():
                    for values in panels.values():
                        for counts in values.values():
                            counts["count"]*=2;counts["total"]*=2
            elif case=="raw_examples":request["development"]["examples"]=["untrusted text"]
            pin="0"*64 if case=="hash" else author.request_sha256(request)
            with self.subTest(case=case),self.assertRaises(ValueError):
                author.author_curriculum(self.root/case,request=request,expected_request_sha256=pin)
            self.assertFalse((self.root/case/"intent.json").exists())
        self.transport.assert_not_called()

    def test_local_digest_required_before_and_after_and_no_source_drift(self):
        for i,before,after,expected in ((0,"0"*64,"3"*64,1),(1,"3"*64,"0"*64,4)):
            request=self.request();calls=self.service(request,before=before,after=after)
            result=self.invoke(request,str(i))["result"]
            self.assertEqual(result["outcome"],"procedural_fallback");self.assertEqual(len(calls),expected)
        request=self.request();calls=self.service(request)
        changed=deepcopy(request["source_sha256"]);changed["experiments/definition_tutor_author.py"]="0"*64
        with mock.patch.object(author,"source_hashes",side_effect=[request["source_sha256"],changed]):
            with self.assertRaisesRegex(RuntimeError,"source changed") as raised:self.invoke(request,"drift")
        self.assertFalse(raised.exception.tutor_report["teacher_cost"]["physical_teacher_work_unknown"])
        self.assertTrue((self.root/"drift"/"response.json").exists())
        before=len(calls)
        with self.assertRaises(author.UnknownTutorRequest):self.invoke(request,"drift")
        self.assertEqual(len(calls),before)

    def test_compact_six_skill_prompt_preserves_only_aggregate_counts(self):
        request=self.request()
        value=author.validate_request(request,expected_sha256=author.request_sha256(request))
        payload=author._chat_payload(value);text="".join(m["content"] for m in payload["messages"])
        self.work["prompt_content_bytes"]=len(text.encode())
        self.work["prompt_wire_bytes"]=len(author.transport._canonical_json(payload).encode())
        self.assertLessEqual(len(text.encode()),3800)
        self.assertNotIn(request["parent"]["weights_sha256"],text)
        self.assertNotIn(request["contract"]["catalogue_manifest_sha256"],text)
        self.assertNotIn("source_sha256",text)
        self.assertEqual(payload["options"]["num_predict"],256)
        self.assertNotIn("prefixItems",json.dumps(payload["format"]))
        field=next(k for k in request["procedural_recipe"] if k!="schema")
        wire=payload["format"]["properties"][field]
        self.assertEqual((wire["minItems"],wire["maxItems"]),(6,6))
        self.assertEqual(set(wire["items"]["enum"]),set(request["contract"]["skills"]))
        content=json.loads(payload["messages"][1]["content"])
        self.assertEqual(content["development"]["panels"],list(author.PANELS))
        self.assertEqual(content["development"]["current"][0][0],[32,32,320,320,320,320,64,64,64,64])
        self.assertEqual(set(content),{"teaching","procedural_recipe","development"})
        large=deepcopy(request)
        for snap in ("current","reference"):
            for families in large["development"][snap]["panels"].values():
                for values in families.values():
                    for key,counts in values.items():
                        n=10**17*(1 if key=="all_query_pair_both" else 10 if key.startswith("known") else 2)
                        counts.update(count=n,total=n)
        with self.assertRaisesRegex(ValueError,"exceeds3800"):self.invoke(large,"oversize")
        self.assertFalse((self.root/"oversize"/"intent.json").exists())
        self.transport.assert_not_called()

    def test_http_error_original_bytes_bounded_and_durable_without_retry(self):
        captured=[]
        for body in (b'{"error":"fixture unsupported schema"}',b'x'*(author.MAX_ERROR_BYTES+19)):
            stream=io.BytesIO(body)
            response=mock.Mock(status=400)
            response.getheader.return_value=str(len(body))
            response.read1.side_effect=stream.read
            connection=mock.Mock(sock=None)
            connection.getresponse.return_value=response
            with mock.patch.object(author.http.client,"HTTPConnection",return_value=connection) as constructor:
                self.work["fake_http_connections"]+=1
                with self.assertRaises(author.LocalHTTPError) as caught:
                    WIRE_REQUEST("/api/chat",{"model":"fixture:tiny"},time.monotonic()+10)
                constructor.assert_called_once()
                self.assertEqual(constructor.call_args.args,("127.0.0.1",11434))
                connection.close.assert_called_once()
            detail=caught.exception.receipt
            self.assertEqual(detail["status"],400)
            self.assertEqual(base64.b64decode(detail["body_base64"]),body[:author.MAX_ERROR_BYTES])
            self.assertEqual(detail["truncated"],len(body)>author.MAX_ERROR_BYTES)
            captured.append(caught.exception)
        request=self.request();calls=self.service(request,error=captured[0])
        with self.assertRaises(author.UnknownTutorRequest) as raised:self.invoke(request,"http_error")
        cost=raised.exception.tutor_report["teacher_cost"]
        event=cost["api_events"][-1]
        self.assertEqual(event["status"],"http_error")
        path=self.root/"http_error"/event["error_file"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),event["error_file_sha256"])
        self.assertEqual(json.loads(path.read_text())["body_base64"],captured[0].receipt["body_base64"])
        self.assertTrue(cost["physical_teacher_work_unknown"])
        self.assertEqual(cost["chat_completions"],0)
        before=len(calls)
        with self.assertRaises(author.UnknownTutorRequest):self.invoke(request,"http_error")
        self.assertEqual(len(calls),before)

    def test_wrong_request_binding_and_expired_deadline_do_not_call(self):
        request=self.request(mode="withdrawn");result=self.invoke(request)
        with self.assertRaises(ValueError):
            author.load_result(self.root/"slot"/"result.json",expected_sha256=result["result_sha256"],
                               expected_request_sha256="0"*64)
        with self.assertRaises(TimeoutError):self.invoke(self.request(),"expired",deadline=0)
        self.assertFalse((self.root/"expired"/"intent.json").exists())
        self.transport.assert_not_called()


if __name__=="__main__":
    unittest.main()
